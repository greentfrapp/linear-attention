"""
Task 5 - Translation
Demonstration of full Transformer model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import string
import codecs
import regex
import json
from absl import flags
from absl import app
import seaborn
import matplotlib.pyplot as plt
from matplotlib import gridspec

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("load", False, "Resume training from saved model")
flags.DEFINE_bool("plot", False, "Plot attention heatmaps")
flags.DEFINE_bool("generate", False, "Generate")
flags.DEFINE_bool("eval", False, "Evaluate NLL")

# Training parameters
flags.DEFINE_integer("steps", 50000, "Number of training steps")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_integer("save_every", 50, "Interval between saving model")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 64, "Training batchsize per step")
flags.DEFINE_float("lr", 1e-4, "Learning rate")

# Model parameters
flags.DEFINE_integer("heads", 4, "Number of heads for multihead attention")
flags.DEFINE_integer("enc_layers", 1, "Number of self-attention layers for encodings")
flags.DEFINE_integer("dec_layers", 6, "Number of self-attention layers for encodings")
flags.DEFINE_integer("hidden", 128, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 20, "Maximum input length from toy task")
flags.DEFINE_integer("line", None, "Line to test")


class Task(object):
	
	def __init__(self):
		self.en_file = "data/train.tags.de-en.en"
		self.en_samples = self.get_samples(self.en_file)
		self.rand_en = np.random.RandomState(1)
		self.n_samples = len(self.en_samples)
		self.en_dict = json.load(open("data/en_dict.json", 'r', encoding='utf-8'))
		self.en_vocab_size = len(self.en_dict)
		self.idx = 0

	def get_samples(self, file):
		text = codecs.open(file, 'r', 'utf-8').read().lower()
		text = regex.sub("<.*>.*</.*>\r\n", "", text)
		text = regex.sub("[^\n\s\p{Latin}']", "", text)
		samples = text.split('\n')
		return samples

	def embed(self, sample, dictionary, max_len=20, sos=False, eos=False):
		sample = sample.split()[:max_len]
		while len(sample) < max_len:
			sample.append('<PAD>')
		if sos:
			tokens = ['<START>']
		else:
			tokens = []
		tokens.extend(sample)
		if eos:
			tokens.append('<PAD>')
		idxs = []
		for token in tokens:
			try:
				idxs.append(dictionary.index(token))
			except:
				idxs.append(dictionary.index('<UNK>'))
		idxs = np.array(idxs)
		return np.eye(len(dictionary))[idxs]

	def next_batch(self, batchsize=64, max_len=20, idx=None):
		start = self.idx
		if idx is not None:
			start = idx
		end = start + batchsize
		if end > self.n_samples:
			end -= self.n_samples
			en_minibatch_text = self.en_samples[start:]
			self.rand_en.shuffle(self.en_samples)
			en_minibatch_text += self.en_samples[:end]
		else:
			en_minibatch_text = self.en_samples[start:end]
		self.idx = end
		en_minibatch_in = []
		en_minibatch_out = []
		for sample in en_minibatch_text:
			en_minibatch_in.append(self.embed(sample, self.en_dict, max_len=max_len, sos=True))
			en_minibatch_out.append(self.embed(sample, self.en_dict, max_len=max_len, eos=True))
		return np.array(en_minibatch_in), np.array(en_minibatch_out)

	def prettify(self, sample, dictionary, probabilistic=False):
		if probabilistic:
			idxs = []
			for dist in sample:
				idx = np.random.choice(np.arange(len(dist)), p=dist)
				idxs.append(idx)
			idxs = np.array(idxs)
		else:
			idxs = np.argmax(sample, axis=1)
		return " ".join(np.array(dictionary)[idxs])


class AttentionModel(object):

	def __init__(self, sess, en_vocab_size, max_len=20, hidden=64, name="Translate", dec_layers=6, learning_rate=1e-4):
		super(AttentionModel, self).__init__()
		self.sess = sess
		self.max_len = max_len
		self.en_vocab_size = en_vocab_size
		self.hidden = hidden
		self.name = name
		self.dec_layers = dec_layers
		self.learning_rate = learning_rate
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		self.input = tf.placeholder(
			shape=(None, self.max_len+1, self.en_vocab_size),
			dtype=tf.float32,
			name="input",
		)

		self.labels = tf.placeholder(
			shape=(None, self.max_len+1, self.en_vocab_size),
			dtype=tf.float32,
			name="labels",
		)

		pos_enc = tf.Variable(
			initial_value=tf.zeros((1, self.max_len+1, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="positional_coding"
		)

		# Embed inputs to hidden dimension
		input_emb = tf.layers.dense(
			inputs=self.input,
			units=self.hidden,
			activation=None,
			name="input_embedding",
		)

		# Add positional encodings
		embedding = input_emb + pos_enc

		for i in np.arange(self.dec_layers):
			# Decoder Self-Attention
			embedding, _ = self.multihead_attention(
				query=embedding,
				key=embedding,
				value=embedding,
				mask=True,
			)
			# Decoder Dense
			dense = tf.layers.dense(
				inputs=embedding,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="decoder_layer{}_dense1".format(i + 1)
			)
			embedding += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="decoder_layer{}_dense2".format(i + 1)
			)
			embedding = tf.contrib.layers.layer_norm(embedding, begin_norm_axis=2)

		decoding = tf.layers.dense(
			inputs=embedding,
			units=self.en_vocab_size,
			activation=None,
			name="decoding",
		)

		self.logits = decoding
		self.softmax = tf.nn.softmax(self.logits)
		self.predictions = tf.argmax(self.logits, axis=2)
		# self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.expand_dims(self.labels[:, -1], axis=0), logits=tf.expand_dims(self.logits[:, -1], axis=0)))
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	def attention(self, query, key, value, mask=False):
		output = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[2], tf.float32) ** 0.5)
		attention_weights = tf.nn.softmax(output)
		if mask:
			attention_weights = tf.matrix_band_part(attention_weights, -1, 0)
			attention_weights /= tf.reduce_sum(attention_weights, axis=2, keep_dims=True)
		weighted_sum = tf.matmul(attention_weights, value)
		output = weighted_sum + query
		output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
		return output, attention_weights

	def multihead_attention(self, query, key, value, h=4, mask=False):
		W_query = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_key = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_value = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_output = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		multi_query = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(query, [-1, self.hidden]), W_query), [-1, 1, tf.shape(query)[1], h, int(self.hidden/h)]), axis=3), axis= 1)
		multi_key = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(key, [-1, self.hidden]), W_key), [-1, 1, tf.shape(key)[1], h, int(self.hidden/h)]), axis=3), axis= 1)
		multi_value = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(value, [-1, self.hidden]), W_value), [-1, 1, tf.shape(value)[1], h, int(self.hidden/h)]), axis=3), axis= 1)
		dotp = tf.matmul(multi_query, multi_key, transpose_b=True) / (tf.cast(tf.shape(multi_query)[-1], tf.float32) ** 0.5)
		attention_weights = tf.nn.softmax(dotp)

		if mask:
			attention_weights = tf.matrix_band_part(attention_weights, -1, 0)
			attention_weights /= tf.reduce_sum(attention_weights, axis=3, keep_dims=True)

		weighted_sum = tf.matmul(attention_weights, multi_value)
		weighted_sum = tf.concat(tf.unstack(weighted_sum, axis=1), axis=-1)
		
		multihead = tf.reshape(tf.matmul(tf.reshape(weighted_sum, [-1, self.hidden]), W_output), [-1, tf.shape(query)[1], self.hidden])
		output = multihead + query
		output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
		return output, attention_weights
		

	def save(self, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(self.sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(self.sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))
		return ckpt

def main(unused_args):

	if FLAGS.train:
		tf.gfile.MakeDirs(FLAGS.savepath)
		with tf.Session() as sess:
			task = Task()
			model = AttentionModel(
				sess=sess,
				en_vocab_size=task.en_vocab_size,
				max_len=FLAGS.max_len,
				hidden=FLAGS.hidden,
				dec_layers=FLAGS.dec_layers,
				learning_rate=FLAGS.lr,
			)
			if FLAGS.load:
				ckpt = model.load(FLAGS.savepath)
				step = int(ckpt.split("ckpt-")[-1]) + 1
			else:
				sess.run(tf.global_variables_initializer())
				step = 1
			for i in np.arange(FLAGS.steps):
				minibatch_dec_in, minibatch_dec_out = task.next_batch(batchsize=FLAGS.batchsize, max_len=FLAGS.max_len)
				feed_dict = {
					model.input: minibatch_dec_in,
					model.labels: minibatch_dec_out,
				}
				_, loss = sess.run([model.optimize, model.loss], feed_dict)
				if (i + step) % FLAGS.save_every == 0:
					model.save(FLAGS.savepath, global_step=i + step)
				if (i + step) % FLAGS.print_every == 0:
					print("Iteration {} - Loss {:.3f}".format(i + step, loss))
			print("Iteration {} - Loss {:.3f}".format(i + step, loss))
			print("Training complete!")
			model.save(FLAGS.savepath, global_step=i + step, verbose=True)

	elif FLAGS.generate:
		with tf.Session() as sess:
			task = Task()
			model = AttentionModel(
				sess=sess,
				en_vocab_size=task.en_vocab_size,
				max_len=FLAGS.max_len,
				hidden=FLAGS.hidden,
				dec_layers=FLAGS.dec_layers,
				learning_rate=FLAGS.lr,
			)
			model.load(FLAGS.savepath)
			output = " most of the"
			for i in np.arange(FLAGS.max_len):
				feed_dict = {
					model.input: [task.embed(output, task.en_dict, max_len=FLAGS.max_len, sos=True)]
				}
				distribution = sess.run(model.softmax, feed_dict)
				output += " " + task.prettify(distribution[0], task.en_dict, probabilistic=True).split()[i]
			print("\nOutput: \n{}".format(output))

	elif FLAGS.eval:
		with tf.Session() as sess:
			task = Task()
			model = AttentionModel(
				sess=sess,
				en_vocab_size=task.en_vocab_size,
				max_len=FLAGS.max_len,
				hidden=FLAGS.hidden,
				dec_layers=FLAGS.dec_layers,
				learning_rate=FLAGS.lr,
			)
			model.load(FLAGS.savepath)
			dec_in, dec_out = task.next_batch(batchsize=1, max_len=FLAGS.max_len, idx=FLAGS.line)
			feed_dict = {
				model.input: [task.embed(" most of connected of connected ocean water", task.en_dict, sos=True)]
				# model.input: dec_in,
			}
			distribution = sess.run(model.softmax, feed_dict)
			probabilities = distribution * dec_out
			negativell = -np.log(np.prod(np.sum(probabilities, axis=2)[0])) / (FLAGS.max_len + 1)
			print(negativell)

	elif FLAGS.test:
		with tf.Session() as sess:
			task = Task()
			model = AttentionModel(
				sess=sess,
				en_vocab_size=task.en_vocab_size,
				de_vocab_size=task.de_vocab_size,
				max_len=FLAGS.max_len,
				hidden=FLAGS.hidden,
				enc_layers=FLAGS.enc_layers,
				dec_layers=FLAGS.dec_layers,
				heads=FLAGS.heads,
				learning_rate=FLAGS.lr,
			)
			model.load(FLAGS.savepath)
			samples, _, truth = task.next_batch(batchsize=1, max_len=FLAGS.max_len, idx=FLAGS.line)
			print("\nInput : \n{}".format(regex.sub("\s<PAD>", "", task.prettify(samples[0], task.de_dict))))
			print("\nTruth : \n{}".format(regex.sub("\s<PAD>", "", task.prettify(truth[0], task.en_dict))))

			output = ""
			for i in np.arange(FLAGS.max_len):
				feed_dict = {
					model.enc_input: samples,
					model.dec_input: [task.embed(output, task.en_dict, sos=True)],
				}
				predictions, attention = sess.run([model.logits, model.attention], feed_dict)
				output += " " + task.prettify(predictions[0], task.en_dict).split()[i]
			print("\nOutput: \n{}".format(regex.sub("\s<PAD>", "", task.prettify(predictions[0], task.en_dict))))

			if FLAGS.plot:
				fig = plt.figure(figsize=(10, 10))
				gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.5)
				x_labels = regex.sub("\s<PAD>", "", task.prettify(samples[0], task.de_dict)).split()
				y_labels = regex.sub("\s<PAD>", "", task.prettify(predictions[0], task.en_dict)).split()
				for i in np.arange(4):
					ax = plt.Subplot(fig, gs[i])
					seaborn.heatmap(
						data=attention[0][i, :len(y_labels), :len(x_labels)],
						xticklabels=x_labels,
						yticklabels=y_labels,
						cbar=False,
						ax=ax,
					)
					ax.set_title("Head {}".format(i))
					ax.set_aspect('equal')
					for tick in ax.get_xticklabels(): tick.set_rotation(90)
					for tick in ax.get_yticklabels(): tick.set_rotation(0)
					fig.add_subplot(ax)
				plt.show()


if __name__ == "__main__":
	app.run(main)
