---
layout: post
title: "Data preprocessing at scale with tensorflow transform"
categories: machine-learning
author:
- Nigama Vykari
meta: "Springfield"
---

Data preprocessing is a necessary step before applying any machine learning algorithms to your data. It plays a huge role in accuracy since the data is primarily noisy, inconsistent and has missing values, or even false values. The preprocessing step usually occurs after data cleaning and ESA to understand your dataset better. You can think of data preprocessing as a prerequisite for modeling. Once you understand the data, you will have a better idea of how to model it. Most ML models need numerical inputs. So if your data has categorical values, there is a requirement to transform them.

In this article, we will learn about implementing data preprocessing at scale with Tensorflow and the challenges of production-ready ML models.

#### Machine learning in production

Machine learning in production is usually expected to be 80% model building and 20% other production stuff like data mining, preprocessing, etc. However, it is usually the other way around. It’s a challenge to maintain the exact outcomes of feature engineering during model training and deployment.

The reasons that cause these problems mostly have to do with data collection or system alignments. Issues like bad log data, loss of network connectivity, and software updates can be dealt with quickly since we know the challenge and monitor our systems. But if we think long term, we need to consider that trends are constantly changing, which brings changes to the data, leading to new problems. For this reason, we need to “understand” our model inside out so that we can make sure that mispredictions do not cost us money.

#### Challenges of preprocessing data at scale

Applying ML models in real-time requires a lot of preprocessing effort. This means converting between formats and performing various numerical operations. However, there are two main challenges that need to be addressed if we want to deploy a successful model:

Training-serving skew: The data and the code is seen at the serving stage are usually different from what is used to train the model. This can lead to a decrease in production quality or complete failure of the model.
Converting raw inputs to features is another challenge while working with unstructured data. Since machine learning models cannot directly work with text, we need to tokenize them and then use them as inputs.

These challenges can be solved using Tensorflow Transform, which will be covered later in the article.

The lifecycle of any machine learning pipeline consists of the following steps:

- Data ingestion
- Data validation
- Transformation
- Training / Estimating the model
- Model analysis
- Model Serving

This article will only discuss implementing the data preprocessing part with TensorFlow Extended (TFX). The TFX tool is a very effective way to overcome the training serving skew, and it provides many options with each step in the process. We will build a machine learning pipeline by taking the IMDB dataset as an example.

#### Data ingestion

The pipeline starts with data ingestion which means trying to include our data to build a production-ready model. Since we are working with TFX, it provides us with a component called ExampleGen that we usually implement at the beginning of the pipeline.

But before we implement that, we need to set up our paths and import the necessary libraries.

**Note:** If you are using Google Colab, make sure to restart the runtime after installing TFX and before implementing this step to avoid any errors.

```python
import os
import pprint
import tempfile
import urllib
import shutil

from pathlib import Path

import absl
import tensorflow as tf
import tensorflow_datasets as tfds
import tfx

from tfx.components import ImportExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.utils.dsl_utils import external_input

%load_ext tfx.orchestration.experimental.interactive.notebook_extensions.skip
```

You will understand each component in detail as we build this pipeline.

Moving forward, we need to set up directories like `TFRecords` to store the files generated in the future. This step is straightforward and easy.

``` python
data_dir = "./data"
builder = tfds.builder('imdb_reviews', data_dir=data_dir)
builder.download_and_prepare()
tfrecord_dir = os.path.join(data_dir, 'tfrecords')

if not os.path.exists(tfrecord_dir):
    os.mkdir(tfrecord_dir)

if not os.listdir(tfrecord_dir):
    data_path = Path(os.path.join(data_dir, "imdb_reviews")).glob("**/*imdb_reviews-t*")
    records = [str(i) for i in data_path]
    if records:
        for record in records:
            filename = os.path.basename(record)
            to_location = os.path.join(tfrecord_dir, filename)
            shutil.move(record, to_location)
```

Here, we create an `InteractiveContext` using default parameters. This will make use of a temporary directory with an ML metadata database instance.
To use your own pipeline root or database, the optional properties `Pipeline_root` and `metadata_connection_config` may be passed to `InteractiveContext`. Calls to `InteractiveContext` does not work outside of colab, since the component is created specially to make the notebook interactive.

```context = InteractiveContext(pipeline_root="./nlp_tfx/")```

In the next step, we create a data ingestion component from ExampleGen using `ImportExampleGen`. It creates a way of providing data to all components that are using TFDV(TensorFlow Data Validation). It generates training and validation datasets for you by pointing your data to the file containing the `TFRecords`.

This is similar to `train_test_split` done with SciKit-learn. TFX creates training and validation datasets even when the data is huge and is scattered in different files.

```python
examples = external_input(tfrecord_dir)

#create TFX data ingestion component
example_gen = ImportExampleGen(input=examples)

#run TFX data ingestion component
context.run(example_gen)
```

#### Data validation

The next step in the pipeline is data validation. This step is important to:

- understand our data,
- generate a schema, and
- find unnecessary values.

Since TFX is an end-to-end machine learning platform, we can rely on TFDV for this step. The component called StatisticsGen by TFDV takes away the burden of exploratory data analysis by taking in the output of any given ExampleGen as input and forming a small pipeline. Take a look at the code below for a better understanding.

```python
#create TFX StatisticsGen component and pass output of ExampleGen as input
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

#run the component
context.run(statistics_gen)
```

The outcomes are stored in a temporary file containing all the numbers for training and validation datasets and `tf-records`.

It is now time to visualize our dataset to get a better understanding of the statistics.

```python
# visualize the results of StatisticsGen
context.show(statistics_gen.outputs['statistics'])
```

To generate statistics on both training and evaluation datasets, we use another TFX component called StatisticsGen. To acquire the output directories of StatisticsGen and inspect them, we need to implement the following.

```python
# get the output directory of the statisticsGen component
statistics_dir = statistics_gen.outputs['statistics'].get()[0].uri
# inspect directory
! ls {statistics_dir}
```

In the output, you will notice that eval and train are the directories we need.

In the next step, the results from StatisticsGen are given as input to the next component called SchemaGen. The reason for using this component is to allow the data to parse more easily. It helps us discover any anomalies, outliers, or errors in our dataset.

We can visualize the generated schema as a table after it finishes running. The output of the following code creates a `.pbtxt` file with the schema.

I recommend implementing each step separately as given here to observe the outputs clearly.

```python
# create TFX SchemaGen Component and pass the output of StatisticsGen as input to the component
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=True)

# run the component
context.run(schema_gen)
```

Output: Schema is generated and we can see the component mapping after the execution.

```python
# the result shows the generated schema which can be used to validate new inputs
context.show(schema_gen.outputs['schema'])
!cat {schema_gen.outputs['schema']._artifacts[0].uri + "/schema.pbtxt"}
```

Output: the result shows the generated schema which can be used to validate new inputs.

The next component we use in the pipeline is called `ExampleValidator`. It combines the outputs of `StatisticsGen` and `SchemaGen` to check for any outliers or false values.

```python
# create TFX ExampleValidator Component and pass the output of StatisticsGen and SchemaGen as inputs to the component
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])

# run the TFX component
context.run(example_validator)

# inspect if any anomalies are present
context.show(example_validator.outputs['anomalies'])
```

You will see that there are no anomalies at this point.

The two most essential data preprocessing steps are feature selection, and feature extraction collectively referred to as feature engineering. This directly impacts the model performance during the production stage. Most businesses that use machine learning train their models on platforms like Google Cloud or Apache Spark due to a lack of computing power to host large-scale data sets. Hence, there is a high chance of training-serving skew leading to the failure of the production model.

This is where TensorFlow Transform `(tf.Transform)` provides us with a solution to make sure there is consistency in training and serving environments.

#### What is TensorFlow Transform

Tensorflow Transform is a library from TensorFlow which allows you to express the preprocessing function data that contains transformations that require a full pass of your data. It will automatically run a data pre-processing graph to compute those statistics. It is a structured way of analyzing and transforming big data sets.

#### Why use TensorFlow Transform?

As discussed earlier, it is entirely possible that training and serving code are entirely different. Tensorflow Transform acts as a bridge between the training and serving code to make easy transformations and utilize the same code base for both purposes.

Simply put, `tf.Transform` eliminates the disadvantage of training serving skew by applying the same transformations. Therefore, at the time of serving, we just need to fit in the raw data and all transformations are done as a part of the TensorFlow graph.

Now, let us continue building our pipeline and see how to use Tensorflow Transform.

To make use of the Tensorflow Transform component, we need to perform the following steps:

- Define a preprocessing function (preprocessing_fn) in a new file. Here, it is called `transform.py`
- Define `tf.Transform` component to call the preprocessing function.

We can say that `preprocessing_fn` is the entry point into TF Transform.To convert strings into numerical values, we need a helper function. The transformations used to convert the inputs into features are:

- Creating two helper functions: `text_standardization` and `transform`
- Once the features are transformed, we output them using a dictionary.

```python
%%writefile transform.py
import tensorflow as tf
import tensorflow_text as tftext
import tensorflow_hub as hub
BERT_TFHUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
SEQ_LEN = 150
# load tensorflow hub bert tokenize layer
def load_bert_layer(model_url=BERT_TFHUB_URL):
   # Load the pre-trained BERT model as layer in Keras
   bert_layer = hub.KerasLayer(
                         handle=model_url,
                         trainable=True
                         )
   return bert_layer
def preprocessing_fn(inputs):
   """ function to be called by Transform Component"""
   vocab_file_path = load_bert_layer()
                             .resolved_object
                             .vocab_file.asset_path
   bert_tokenizer = tftext.BertTokenizer(
                           vocab_lookup_table=vocab_file_path,
                           token_out_type=tf.int64)
def text_standardization(input_data):
       """ convert to lower case and remove html """
       lowercase = tf.strings.lower(input_data)
       stripped_html = tf.strings.regex_replace(
                                  lowercase,
                                  "<br />", " ")
       return stripped_html
def transform(text, sequence_length=SEQ_LEN):
       """ standardize and tokenize text """
      
       # strip text and tokenize
       text = text_standardization(text)
       tokens = bert_tokenizer.tokenize(text)
       pad_id = tf.constant(0, dtype=tf.int64)
      
       # format tensor for batch
       tokens = tokens.merge_dims(1, 2)
       tokens = tokens[:, :sequence_length]
       tokens = tokens.to_tensor(default_value=pad_id)
       pad = sequence_length - tf.shape(tokens)[1]
       tokens = tf.pad(tokens,
                       [[0, 0], [0, pad]],
                       constant_values=pad_id)
       return tf.reshape(tokens, [-1, sequence_length])
# apply transformations
   text = inputs['text']
   text = tf.squeeze(text, axis=1)
   text_transformed = transform(text)
   label = inputs['label']
return {
       "sentence_xf": text_transformed,
       "label": label
   }

from tfx.components import Transform
# create the Transform Component
transform = Transform(
   examples=example_gen.outputs['examples'],
   schema=schema_gen.outputs['schema'],
   module_file=os.path.abspath("transform.py")
)
context.run(transform)
```

The transform component takes in `example_gen`, `schema`, and `transform.py` as inputs:

- Example_gen creates TFRecords as outputs. The raw inputs from `TFRecord` is parsed with the help of the schema that we generated.
- The file containing the `preprocessing_fn` will be called to make transformations of inputs to features.
- The preprocessing graph, metadata, and the converted dataset are saved to the disk.

#### Steps in preprocessing the data

When `tf.Transform`  is implemented, it preprocesses the data with the help of Apache Beam in two phases.

- **Analyze phase:** The statistics required for the transformations on your model get computed on the training data.
- **Transform phase:** The statistics generated are used to process the training data with instance-level operations.

#### Implementing Apache Beam

Apache Beam is a popular parallel computation framework that proposes some significant benefits over similar frameworks. It is an open-source SDK for the Dataflow model, which explains the Google cloud data processing workflow.

So far, all the previous processing frameworks have tried to optimize along with the correctness, latency, and cost dimensions. For example, to ensure that the data is complete, you will probably wait a bit more before processing to ensure all the necessary data has arrived, which causes high latency. You might decide to start the processing early to minimize latency, resulting in incomplete data and increased costs. The problem with these frameworks is that they all assume the input data will be complete at some point.

The solution provided by the Apache Beam is a unified model which incorporates the facts that we might never know for sure whether or not we’ve seen all our data. It is unified because it doesn’t differentiate bounded and unbounded datasets or streaming and batch processing. The Dataflow model is infrastructure agnostic, which means it can run on different execution engines, making the choice of execution engine a practical decision.

In this section, we utilize Apache Beam to analyze and transform the raw inputs into features and create feature specs to create a schema.

```python
# Define feature spec for parsing features
FEATURE_SPEC = {
   "text": tf.io.FixedLenFeature([1], tf.string),
   "label": tf.io.FixedLenFeature([1], tf.int64)
}
# Create metadata schema from feature spec
RAW_DATA_METADATA = tft.tf_metadata.dataset_metadata.
                       DatasetMetadata(
                            tft.tf_metadata.dataset_schema
                            .schema_utils.
                            schema_from_feature_spec(
                                               FEATURE_SPEC
                                               )
                                          )
```

In the next step, we will define the beam pipeline and save the results to the disk.

```python
with beam.Pipeline() as pipeline:
   with tft_beam.impl.Context(temp_dir=MODEL_DIRECTORY):
RAW_DATA_TRAIN = (
       pipeline
       | "prep_train_input" >> ReadTFRecordAndPrepInput(
                                          PATH_TO_TFRECORD_TRAIN)
       )
```

To transform the raw inputs, we use a function called `AnalyzeAndTransformDataset`. After the data is transformed, we get in return:

- a transformation function,
- a metadata schema and,
- the transformed data.

The generated schema could be used to encode the training data and save it to the disk.

```python
# analyze and transform training dataset
    transformed_dataset, transform_fn = (
           (RAW_DATA_TRAIN, RAW_DATA_METADATA)
           | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
    )
transformed_data, transformed_metadata = transformed_dataset
    transformed_data_coder = tft.coders.ExampleProtoCoder(
                                       transformed_metadata.schema)
# Serialize and write train data to TFRecord
     _ = (
     transformed_data
     | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
     | 'WriteTrainData' >> beam.io.WriteToTFRecord(
                              os.path.join(
                                 MODEL_DIRECTORY,
                                 TRANSFORMED_TRAIN_DATA_FILEBASE))
     )
```

Now, we convert the dataset. We can make use of the transformation function from the previous step to analyze the dataset.

```python
RAW_DATA_TEST = (
          pipeline
          | "prep_test_input" >> ReadTFRecordAndPrepInput(
                                            PATH_TO_TFRECORD_TEST)
          )
raw_test_dataset = (RAW_DATA_TEST, RAW_DATA_METADATA)
    transformed_test_dataset = (
          (raw_test_dataset, transform_fn)
          | tft_beam.TransformDataset()
    )
# Don't need transformed data schema, it's the same as before.
    transformed_test_data, _ = transformed_test_dataset
# Serialize and write test data to TFRecord
    _ = (
    transformed_test_data
    | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
    | 'WriteTestData' >> beam.io.WriteToTFRecord(
                              os.path.join(
                                  MODEL_DIRECTORY,
                                  TRANSFORMED_TEST_DATA_FILEBASE))
    )
# write transform graph
    _ = (
    transform_fn
    | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
                                          MODEL_DIRECTORY)
    )

```

If everything runs properly, your transformations are saved and your model is preprocessed and ready for serving. You have now successfully preprocessed your data with the help of TensorFlow Transform.

#### Final thoughts

So far, we have covered everything from challenges in feature engineering to using Tensorflow Transform in terms of data preprocessing. To try the example covered in this article, you can implement it using this Colab Notebook. This is only halfway through the pipeline.

The next step is to focus on serving the model. Thankfully, tools like Tensorflow make our job a little bit easier with these challenging tasks. 