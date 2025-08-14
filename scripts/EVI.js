//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B04", "B08"],  // Blue, Red, NIR bands
      units: "DN"
    }],
    output: {
      bands: 1,
      sampleType: SampleType.FLOAT32
    },
    mosaicking: Mosaicking.ORBIT
  };
}

function updateOutput(outputs, collection) {
  Object.values(outputs).forEach(output => {
    output.bands = collection.scenes.length;
  });
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  const dates = scenes.map(scene => scene.date);
  outputMetadata.userData = { "acquisition_dates": JSON.stringify(dates) };
}

function evaluatePixel(samples) {
  const eviArray = [];
  samples.forEach(sample => {
    const nir = sample.B08;
    const red = sample.B04;
    const blue = sample.B02;
    const evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1);
    eviArray.push(evi);
  });
  return eviArray;
}
