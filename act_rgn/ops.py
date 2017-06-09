import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from ipdb import set_trace as brk

def actrgn_rnn_decoder(decoder_inputs,
                initial_state,initial_attn_output,
                cell,attn_dim,lstm_dim,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "actrgn_rnn_decoder"):
    state = initial_state
    output = initial_attn_output
    
    outputs = []
    prev = None
    w_l = variable_scope.get_variable(name='lstm_to_attn_w',shape=[lstm_dim,attn_dim],dtype=tf.float32)
    b_l = variable_scope.get_variable(name='lstm_to_attn_b',shape=[attn_dim],dtype=tf.float32)
    w_i = variable_scope.get_variable(name='ip_to_attn_w',shape=[attn_dim,attn_dim],dtype=tf.float32)
    b_i = variable_scope.get_variable(name='ip_to_attn_b',shape=[attn_dim],dtype=tf.float32)
    w_f = variable_scope.get_variable(name='attn_to_prob_w',shape=[attn_dim,1],dtype=tf.float32)
    b_f = variable_scope.get_variable(name='attn_to_prob_b',shape=[1],dtype=tf.float32)


    for i, inp in enumerate(decoder_inputs):

      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      
      attn_state = tf.matmul(output,w_l) + b_l
      
      context_state = tf.matmul(inp,tf.tile(tf.expand_dims(w_i,0),[int(inp.shape[0]),1,1])) + b_i
      
      context_state = context_state + tf.expand_dims(attn_state,1)
      context_state = tf.tanh(context_state)

      attn_prob = tf.squeeze(tf.nn.softmax(tf.matmul(context_state,tf.tile(tf.expand_dims(w_f,0),[int(context_state.shape[0]),1,1])) + b_f))

      inp_rnn = tf.reduce_sum(tf.multiply(inp,tf.expand_dims(attn_prob,2)),1)

      output, state = cell(inp_rnn, state)

      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state



def actrgn_attention_decoder(decoder_inputs,
                  initial_state,
                  attention_states,
                  cell,
                  output_size=None,
                  num_heads=1,
                  loop_function=None,
                  dtype=None,
                  scope=None,
                  initial_state_attention=False):
	"""RNN decoder with attention for the sequence-to-sequence model.

	In this context "attention" means that, during decoding, the RNN can look up
	information in the additional tensor attention_states, and it does this by
	focusing on a few entries from the tensor. This model has proven to yield
	especially good results in a number of sequence-to-sequence tasks. This
	implementation is based on http://arxiv.org/abs/1412.7449 (see below for
	details). It is recommended for complex sequence-to-sequence tasks.

	Args:
	decoder_inputs: A list of 2D Tensors [batch_size x input_size].
	initial_state: 2D Tensor [batch_size x cell.state_size].
	attention_states: 3D Tensor [batch_size x attn_length x attn_size].
	cell: core_rnn_cell.RNNCell defining the cell function and size.
	output_size: Size of the output vectors; if None, we use cell.output_size.
	num_heads: Number of attention heads that read from attention_states.
	loop_function: If not None, this function will be applied to i-th output
	  in order to generate i+1-th input, and decoder_inputs will be ignored,
	  except for the first element ("GO" symbol). This can be used for decoding,
	  but also for training to emulate http://arxiv.org/abs/1506.03099.
	  Signature -- loop_function(prev, i) = next
	    * prev is a 2D Tensor of shape [batch_size x output_size],
	    * i is an integer, the step number (when advanced control is needed),
	    * next is a 2D Tensor of shape [batch_size x input_size].
	dtype: The dtype to use for the RNN initial state (default: tf.float32).
	scope: VariableScope for the created subgraph; default: "attention_decoder".
	initial_state_attention: If False (default), initial attentions are zero.
	  If True, initialize the attentions from the initial state and attention
	  states -- useful when we wish to resume decoding from a previously
	  stored decoder state and attention states.

	Returns:
	A tuple of the form (outputs, state), where:
	  outputs: A list of the same length as decoder_inputs of 2D Tensors of
	    shape [batch_size x output_size]. These represent the generated outputs.
	    Output i is computed from input i (which is either the i-th element
	    of decoder_inputs or loop_function(output {i-1}, i)) as follows.
	    First, we run the cell on a combination of the input and previous
	    attention masks:
	      cell_output, new_state = cell(linear(input, prev_attn), prev_state).
	    Then, we calculate new attention masks:
	      new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
	    and then we calculate the output:
	      output = linear(cell_output, new_attn).
	  state: The state of each decoder cell the final time-step.
	    It is a 2D Tensor of shape [batch_size x cell.state_size].

	Raises:
	ValueError: when num_heads is not positive, there are no inputs, shapes
	  of attention_states are not set, or input size cannot be inferred
	  from the input.
	"""
	if not decoder_inputs:
		raise ValueError("Must provide at least 1 input to attention decoder.")
	if num_heads < 1:
		raise ValueError("With less than 1 heads, use a non-attention decoder.")
	if attention_states.get_shape()[2].value is None:
		raise ValueError("Shape[2] of attention_states must be known: %s" %
	                 attention_states.get_shape())
	if output_size is None:
		output_size = cell.output_size

	with variable_scope.variable_scope(
	  scope or "attention_decoder", dtype=dtype) as scope:
		dtype = scope.dtype

	batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
	attn_length = attention_states.get_shape()[1].value
	if attn_length is None:
	  attn_length = array_ops.shape(attention_states)[1]
	attn_size = attention_states.get_shape()[2].value

	# To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
	hidden = array_ops.reshape(attention_states,
	                           [-1, attn_length, 1, attn_size])
	hidden_features = []
	v = []
	attention_vec_size = attn_size  # Size of query vectors for attention.
	for a in xrange(num_heads):
	  k = variable_scope.get_variable("AttnW_%d" % a,
	                                  [1, 1, attn_size, attention_vec_size])
	  hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
	  v.append(
	      variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

	state = initial_state

	def attention(query):
	  """Put attention masks on hidden using hidden_features and query."""
	  ds = []  # Results of attention reads will be stored here.
	  if nest.is_sequence(query):  # If the query is a tuple, flatten it.
	    query_list = nest.flatten(query)
	    for q in query_list:  # Check that ndims == 2 if specified.
	      ndims = q.get_shape().ndims
	      if ndims:
	        assert ndims == 2
	    query = array_ops.concat(query_list, 1)
	  for a in xrange(num_heads):
	    with variable_scope.variable_scope("Attention_%d" % a):
	      y = linear(query, attention_vec_size, True)
	      y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
	      # Attention mask is a softmax of v^T * tanh(...).
	      s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
	                              [2, 3])
	      a = nn_ops.softmax(s)
	      # Now calculate the attention-weighted vector d.
	      d = math_ops.reduce_sum(
	          array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
	      ds.append(array_ops.reshape(d, [-1, attn_size]))
	  return ds

	outputs = []
	prev = None
	batch_attn_size = array_ops.stack([batch_size, attn_size])
	attns = [
	    array_ops.zeros(
	        batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
	]
	for a in attns:  # Ensure the second shape of attention vectors is set.
	  a.set_shape([None, attn_size])
	if initial_state_attention:
	  attns = attention(initial_state)
	for i, inp in enumerate(decoder_inputs):
	  if i > 0:
	    variable_scope.get_variable_scope().reuse_variables()
	  # If loop_function is set, we use it instead of decoder_inputs.
	  if loop_function is not None and prev is not None:
	    with variable_scope.variable_scope("loop_function", reuse=True):
	      inp = loop_function(prev, i)
	  # Merge input and previous attentions into one vector of the right size.
	  input_size = inp.get_shape().with_rank(2)[1]
	  if input_size.value is None:
	    raise ValueError("Could not infer input size from input: %s" % inp.name)
	  x = linear([inp] + attns, input_size, True)
	  # Run the RNN.
	  cell_output, state = cell(x, state)
	  # Run the attention mechanism.
	  if i == 0 and initial_state_attention:
	    with variable_scope.variable_scope(
	        variable_scope.get_variable_scope(), reuse=True):
	      attns = attention(state)
	  else:
	    attns = attention(state)

	  with variable_scope.variable_scope("AttnOutputProjection"):
	    output = linear([cell_output] + attns, output_size, True)
	  if loop_function is not None:
	    prev = output
	  outputs.append(output)

	return outputs, state

