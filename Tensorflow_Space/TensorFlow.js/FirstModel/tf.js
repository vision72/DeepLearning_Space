const model = tf.sequential();
model.add(tf.layers.dense({units: 784, inputShape: [1]})) // input-layer
model.add(tf.layers.dense({units: 900, inputShape: [1]})) // second-layer
model.add(tf.layers.dense({units: 900, inputShape: [1]})) // third-layer
model.add(tf.layers.dense({units: 10, inputShape: [1]})) // output-layer
model.compile({optimizer: tf.train.sgd(0.17), loss: "meanSquaredError"})

console.log("I am here");
