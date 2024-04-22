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