# Load the training and testing set of CIFAR10
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.astype('float32')
X_train = X_train/255.

X_test = X_test.astype('float32')
X_test = X_test/255.

y_train = tf.reshape(tf.one_hot(y_train, 10), shape=(-1, 10))
y_test = tf.reshape(tf.one_hot(y_test, 10), shape=(-1, 10))

# Create TensorFlow dataset
BATCH_SIZE = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1024).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

org1 = Organism({
    'a_filter_size': options['a_filter_size'][np.random.randint(len(options['a_filter_size']))],
    'a_include_BN': options['a_include_BN'][np.random.randint(len(options['a_include_BN']))],
    'a_output_channels': options['a_output_channels'][np.random.randint(len(options['a_output_channels']))],
    'activation_type': options['activation_type'][np.random.randint(len(options['activation_type']))],
    'b_filter_size': options['b_filter_size'][np.random.randint(len(options['b_filter_size']))],
    'b_include_BN': options['b_include_BN'][np.random.randint(len(options['b_include_BN']))],
    'b_output_channels': options['b_output_channels'][np.random.randint(len(options['b_output_channels']))],
    'include_pool': options['include_pool'][np.random.randint(len(options['include_pool']))],
    'pool_type': options['pool_type'][np.random.randint(len(options['pool_type']))]
    },
    phase=0)

org2 = Organism({
    'a_filter_size': options['a_filter_size'][np.random.randint(len(options['a_filter_size']))],
    'a_include_BN': options['a_include_BN'][np.random.randint(len(options['a_include_BN']))],
    'a_output_channels': options['a_output_channels'][np.random.randint(len(options['a_output_channels']))],
    'activation_type': options['activation_type'][np.random.randint(len(options['activation_type']))],
    'b_filter_size': options['b_filter_size'][np.random.randint(len(options['b_filter_size']))],
    'b_include_BN': options['b_include_BN'][np.random.randint(len(options['b_include_BN']))],
    'b_output_channels': options['b_output_channels'][np.random.randint(len(options['b_output_channels']))],
    'include_pool': options['include_pool'][np.random.randint(len(options['include_pool']))],
    'pool_type': options['pool_type'][np.random.randint(len(options['pool_type']))]
    },
    phase=0)

org1.build_model()
org2.build_model()

org1.fitnessFunction(train_ds, test_ds)
org2.fitnessFunction(train_ds, test_ds)

print('org1: {:0.2f}'.format(org1.fitness))
print('org2: {:0.2f}'.format(org2.fitness))

org1_2 = org1.crossover(org2)
org1_2.mutation(0.1)
org1_2.chromosome

org1_2.build_model()
org1_2.fitnessFunction(train_ds, test_ds)

print('org1_2: {:0.2f}'.format(org1_2.fitness))