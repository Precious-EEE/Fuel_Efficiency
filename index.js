// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LogisticRegression = require("./multinominal-logistic-regression");
const _ = require("lodash");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["mpg"],
  shuffle: true,
  splitTest: 50,
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);
      if (mpg < 15) {
        return [1, 0, 0]; // low class
      } else if (mpg < 30) {
        return [0, 1, 0]; // medium class
      } else {
        return [0, 0, 1]; // high class
      }
    },
  },
});

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
}); // flatmap removes one level of nested arrays

regression.train();

// regression.predict([[150, 200, 2.223]]).print();

console.log(regression.test(testFeatures, _.flatMap(testLabels))); // 0.8