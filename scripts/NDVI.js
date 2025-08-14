//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08"], units: "DN" }],
    output: { bands: 1, sampleType: SampleType.FLOAT32 },
    mosaicking: Mosaicking.ORBIT
  };
}

function updateOutput(outputs, collection) {
  Object.values(outputs).forEach(output => {
    output.bands = collection.scenes.length;
  });
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  let dates = scenes.map(scene => scene.date);
  outputMetadata.userData = { acquisition_dates: JSON.stringify(dates) };
}

function evaluatePixel(samples) {
  let ndviArray = samples.map(sample =>
    (sample.B08 - sample.B04) / (sample.B08 + sample.B04)
  );
  return ndviArray;
}
