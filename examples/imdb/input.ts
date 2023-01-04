import { Data } from "../../lib";

const VOCAB_SIZE = 10_000;

// custom parsing
const parseToTuples = (file: string) =>
  function* (): Iterable<[number[], number[]]> {
    for (const line of file.split("\n")) {
      const [label, ...encoded] = line.split(" ");
      // TODO libsvm format encoding
      const bow = new Array(VOCAB_SIZE).fill(0);
      encoded
        .map((code) => code.split(":").map(Number))
        .filter(([i]) => i < VOCAB_SIZE)
        .forEach(([i, c]) => {
          bow[i] = c;
        });
      yield [bow, [Number(label) > 5 ? 1 : 0]];
    }
  };

const [train, valid] = Data.pipeline(
  await Bun.file("data/train/labeledBow.feat").text(),
  parseToTuples,
  Data.mapExampleTuple(),
  Data.shuffle(),
  Data.batch(100),
  Data.split(0.92) // 2000 validation examples
);

// export processed data sets to disk for fast training later
Data.toDisk(train, "input/train");
Data.toDisk(valid, "input/valid");
