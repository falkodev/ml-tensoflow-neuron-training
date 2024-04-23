import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'
const INPUTS = TRAINING_DATA.inputs // features (number of bedrooms, number of bathrooms, square footage, etc.)
const OUTPUTS = TRAINING_DATA.outputs // predictions (price of the house)

tf.util.shuffleCombo(INPUTS, OUTPUTS)
const INPUTS_TENSOR = tf.tensor2d(INPUTS)
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS)

const FEATURE_RESULTS = normalize(INPUTS_TENSOR)
console.log('Normalized values ====> ')
FEATURE_RESULTS.NORMALIZED_VALUES.print()
console.log('Min values ====> ')
FEATURE_RESULTS.MIN_VALUES.print()
console.log('Max values ====> ')
FEATURE_RESULTS.MAX_VALUES.print()

INPUTS_TENSOR.dispose()

const model = tf.sequential()
model.add(tf.layers.dense({ inputShape: [2], units: 1 }))
model.summary()
train()

function normalize(tensor, min, max) {
  const result = tf.tidy(function() {
    const MIN_VALUES = min || tf.min(tensor, 0)
    const MAX_VALUES = max || tf.max(tensor, 0)
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES)
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES)
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE)

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES }
  })

  return result
}

async function train() {
  const LEARNING_RATE = 0.01
  model.compile({ optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError' })

  let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    validationSplits: 0.15, // part of the data to be used for validation and not for training
    shuffle: true,
    batchSize: 64,
    epochs: 10 // train data 10 times
  })

  OUTPUTS_TENSOR.dispose()
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose()

  console.log('Average error loss: ' + Math.sqrt(results.history.loss[results.history.loss.length - 1]))

  evaluate()
}

function evaluate() {
  tf.tidy(function () {
    // test wiht a house prediction with 750 sqft and 1 bedroom
    let newInput = normalize(tf.tensor2d([[750, 1]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES)
    let output = model.predict(newInput.NORMALIZED_VALUES)
    output.print()
  })

  FEATURE_RESULTS.MIN_VALUES.dispose()
  FEATURE_RESULTS.MAX_VALUES.dispose()
  model.dispose()

  console.log(tf.memory().numTensors)
}