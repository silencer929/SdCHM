//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B08"],     // Green and NIR bands for water detection
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
  var dates = scenes.map(scene => scene.date);
  outputMetadata.userData = { "acquisition_dates": JSON.stringify(dates) };
}

function evaluatePixel(samples) {
  var ndwiArr = new Array(samples.length);
  samples.forEach((sample, idx) => {
    ndwiArr[idx] = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  });
  return ndwiArr;
}
