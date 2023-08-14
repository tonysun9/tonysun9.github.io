---
layout: post
title: An Intro to Real-Time Machine Learning
subtitle: Preparing Features for Prediction
date: 2023-08-13 00:00:00
description: Preparing features for prediction
# tags: formatting images
# categories: sample-posts
thumbnail: /blog/2023/intro-rtml/rt-prediction.png
---



I’ve noticed a trend in the data processing world to move from processing data in batches to processing data continuously in a stream [1]. Although there are new challenges associated with stream processing [2], teams make the transition to get real-time insights and make real-time decisions. This, in turn, can lead to a better user experience and create a competitive advantage.

Netflix is a premier example of a company that has taken the leap towards real-time data infrastructure [3, 4]. This has enabled Netflix to improve their user experience in a variety of ways, ranging from improving recommendations on the “Trending Now” home screen to quickly resolving playback issues, from rapidly testing changes in production to minimizing downtime of the Netflix service [5, 6, 7, 8].

Meanwhile, the field of machine learning has also grown tremendously in the past decade. Machine learning models have become integral to the services offered by companies in a host of domains, including autonomous driving, dynamic pricing, and fraud detection. However, there are many engineering challenges to deploying these models effectively.

This post resides at the intersection of these two fields: real-time data infrastructure and machine learning [9,10]. This post is primarily intended for data scientists and machine learning engineers who want to gain a better understanding of the **underlying data pipelines to serve features for real-time prediction**. In particular, this post addresses three main questions: 



1. What is real-time machine learning?
2. Why is real-time machine learning important?
3. How do we prepare features to be readily available for querying in a real-time prediction service?


## What is real-time machine learning?

The critical component of _real-time machine learning_ is the use of machine learning models to **make predictions in real-time**. Specifically, a prediction is made through a _synchronous request_ and a response is expected to return immediately – on the order of hundreds of milliseconds, oftentimes less.

Contrast this to making predictions in batch, in which predictions are made on a large volume of data points all at once [11, *]. Predictions are made via an _asynchronous batch job_. I have heard the term _batch machine learning_ to describe this concept, although it doesn’t appear to be highly prevalent [12]. 

Let’s work with a fraud detection example. In this example, a consumer purchases a laptop online. The credit card network, say Visa, tries to detect whether the transaction is fraudulent or not.

Using the real-time machine learning paradigm, a prediction is made in _real-time_. That might look like the following:



1. The transaction is an _event_ relayed by a _message broker_ that in turn triggers a _request_ to a _prediction service_ that predicts whether the transaction is fraudulent or not.
2. The prediction service is responsible for fetching the relevant features from the _feature store_ and passing those features along to the _model endpoint_ to make a prediction. 
3. If the transaction is believed to be fraudulent, the transaction is flagged as such and prevented from going through.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/rt-prediction.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Using the batch machine learning paradigm, predictions are made in _batch_. Here’s how that might happen: 



1. During the day, transactions are accumulated into a _data warehouse_. 
2. Periodically, say nightly, an _orchestrator_ kicks off an asynchronous batch job that processes the data. The job involves extracting raw data from the data warehouse, cleaning and transforming that data into features, and making predictions in _batch_ [13]. 
3. For transactions deemed fraudulent, an alert is raised.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/batch-prediction.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>




### Terminology

In case any of the terms around real-time prediction [14] were confusing, let’s take a moment to clarify some terms in the context of machine learning operations:


#### real-time prediction = online prediction; batch prediction = offline prediction

The terms “real-time prediction” and “online prediction” are often used synonymously. Similarly, “batch prediction” and “offline prediction” are interchangeable. I have a preference for the terms “real-time” and “batch” over their counterparts, “online” and “offline,” respectively, from the clarity they provide on the nature of the prediction process [*].


#### Real-time Prediction Terms

Event Broker: kind of like a post office for software, accepting messages (events) from one part of a system (producers) and delivering them to other parts of the system (consumers) [*].

Feature store: a database that serves as a central repository for storing, managing, and serving machine learning features [*]. 

Prediction Service: the infrastructure and related services set up to use a machine learning model for making predictions [*]. In our example, this includes querying the feature store and passing those features to a model endpoint.

Model Endpoint: the network interface exposed by a deployed machine learning model, typically via a RESTful API, to receive and respond to prediction requests [*].


## Why is real-time machine learning important?

Before we dive into preparing features for real-time prediction, let’s first understand why real-time machine learning is important.

Real-time machine learning is powerful for its ability to help **get real-time insights and make real-time decisions.** This information and these choices are critical for some applications, improve user experience in others, and enable proactive responses in yet others.

In our example, flagging a transaction as fraudulent in real-time and preventing that transaction from going through is far more effective than initiating the chargeback process, which requires multiple steps of verification, dispute, and potential reversal [*]. Given that fraud detection is a $385 billion industry [15], any reduction in financial loss can be of significant magnitude.

Here’s a brief and by no means comprehensive list of some applications that benefit from real-time machine learning:



* Anomaly detection for fraud detection, network security, healthcare monitoring, and quality control.
* Personalized recommendations for marketing, e-commerce, and media and entertainment.
* Real-time decision making for autonomous driving, high-frequency trading, and robotics.

And if you’re reading this post, I’m sure you have a use case in mind that can benefit from real-time machine learning as well.


## How do we prepare features to be readily available for querying in a real-time machine learning pipeline?

Earlier, we characterized a real-time prediction service as a system responsible for **querying for the relevant features from a feature store** and passing those features along to a model endpoint to perform prediction.

The purpose of the feature store is to **reduce the latency of a prediction request**. By pre-computing feature values in advance, we save time that would otherwise be spent calculating these values during the prediction request. This not only makes our prediction service more efficient but also enables us to handle higher volumes of requests in a scalable way [*].

We now turn our focus to the feature preparation component – namely, to **populate the feature store** with pre-computed features such that they can be readily queried when it comes time for real-time prediction.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/feature-store.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


The engineering challenges around preparing features for real-time prediction depend on the types of features we’ll be working with. To break down this question, we’ll first give an example of what a feature store might look like, then cover the different types of features, and finally discuss how to populate them into our feature store.


### Feature Store

Feature stores can vary in complexity, ranging from a simple repository of pre-computed features to a much more intricate system responsible for feature versioning, data quality monitoring, access control, and more.

For our purposes, we will use a simple **key-value store** as our feature store [16]. In our feature store, each key-value pair corresponds to a **computed feature value**. 

Here’s what a key-value pair might look like: the key is a concatenation of the feature name and its corresponding entity value(s). The value represents a computed feature value. 

For example, if the feature is the average transaction amount over the past 3 months, “credit card number” would be a natural choice for the entity of this feature. An example of an entity value could be “1234.” The computed feature value could be $36.08.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/entity-key.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


The _entity_ of a feature relates to how the feature is being computed. One way to think about this is “**feature** of _entity_.” For example, I might care about the **average transaction amount** of a _credit card number_, or I might want to know the **age** of a _customer_. An _entity value_ represents the specific instance of an entity. 

Naturally, a feature can also have more than one entity. For example, the feature “average transaction amount with a given merchant over the past 3 months” would use “credit card number” and “merchant_id” as its entities.


### Real-time Prediction

Real-time prediction involves “querying for the relevant features from a feature store**.” **We just went over “feature store;” now, let’s clarify what we mean by “querying for the relevant features.” 



1. **What** are the features we should query for?
2. **How** do we query for those features from the feature store?


#### What are the features we should query for?

The features we should query for naturally **depend on the model** we intend to use for prediction. Since the model is trained using a certain set of features, it requires those same features to do prediction.

For the sake of our example, let’s say our model was trained with just the following four features (represented as feature name: entity; can be helpful to think “feature _of_ entity”):



1. log(transaction amount): transaction_id
2. customer_age: customer_id
3. \# of transactions in the past 10 minutes: credit card number
4. average transaction amount in the past 3 months: credit card number


#### How do we query for those features from the feature store?

Our goal is to query the feature store (a key-value store in our example) for the features our model requires. For each of those features, we need to construct the corresponding key. In our example, we use a combination of the feature name and the feature’s entity value. The **entity value will depend on the information observed in the event** on which we aim to do prediction.

Say our transaction event looks as follows (represented as field name: value): 



* transaction_id: a0d8
* credit_card_num: 1234
* customer_id: 9092
* …
* transaction_amount: 999.99

For the feature “customer_age,” we can query our feature store for the value corresponding to the key “customer_age_9092,” indicating the age of the customer with customer_id 9092. We do this for each feature.

The exception are features that can be computed using just information in the event. These features, such as “log(transaction amount)”, are computed separately in real-time and do not needed to written to / read from the feature store. We’ll drill in more in the following sections.


### Types of Features

The most common way to categorize features is in terms of discrete (categorical) vs continuous (numerical). However, as a post focused on the engineering challenges of making features readily available for real-time prediction, I’d like to propose two new axes of categorization:



1. **Stateless vs Stateful**
    1. _Stateless features_ can be computed based on the information available in the current event alone. They don't depend on any previous events.
    2. _Stateful features_ require knowledge about previous events or instances. They maintain a “state” from the past [17].
2. **Slow-changing vs Fast-changing**
    3. _Fast-changing features_ are features that can change rapidly, even between events that are close together in time [*].
    4. _Slow-changing features_ are features that don’t change, or change very slowly over time [*].

It may be tempting to think of fast-changing as stateless and slow-changing as stateful. However, this is not necessarily the case. Let’s categorize our previous feature examples:


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/feature-types.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>



1. log(transaction amount): a **stateless and fast-changing** feature. Computing the log of the transaction amount changes with each transaction and requires only the information present in the current event.
2. customer_age: a **stateless and slow-changing** feature. A customer’s age increments slowly. Additionally, it does not depend on any prior events.
3. \# of transactions in the past 10 minutes: a **stateful and fast-changing** feature. To compute this feature, we need to maintain a count of the customer's transactions in the last 10 minutes. Furthermore, the count can rapidly change with every new transaction or when transactions fall out of the 10-minute window.
4. average transaction amount in the past 3 months: a **stateful and slow-changing** feature. Computing this feature requires a record of the customer’s transaction amounts over the past 3 months. Although new transactions and the passing of time may affect this feature, the feature value will likely change slowly.


### Populating the Feature Store

Each of the four types of features requires a different method of computation. We go over each feature type below:


#### Stateless and Fast-Changing

Computing stateless and fast-changing features requires us to **process each event on its own**. For our “log(transaction amount)” feature, this involves extracting the transaction amount from the event and running a function that computes the log of a number.

This can be done through an _event-driven compute service_ such as AWS Lambda.

Stateless and fast-changing features are also known as _real-time_ _feature_s because they are computed on-the-fly, in real-time.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/stateless-fc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Something unique about stateless and fast-changing features is that they are not fetched from the feature store, which only stores pre-computed values. This also means that the entity is not relevant for these features.


#### Stateless and Slow-Changing

Because stateless and slow-changing features are slow-changing by definition, it makes sense to pre-compute these features and load them in the feature store well ahead of prediction time. This can be done through a **batch engine** such as Apache Spark.

This typically involves querying for some data in an external warehouse or something similar. This may involve some data cleaning and transformation, although that may not be necessary, especially for stateless features. For our “age” feature, we would just need to query for the age of each customer_id and store those results in the feature store.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/stateless-sc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


Stateless and slow-changing features can also be considered _batch_ features because they are typically computed in a batch process. The pipeline might look similar to our batch prediction pipeline earlier, except we stop before the model prediction part and instead store the transformed values (features) into our feature store.

Using a batch engine like Apache Spark offers benefits in terms of robustness and efficiency. Spark is fault-tolerant which means it can recover from failures and errors without losing data or functionality. Spark jobs can also scale horizontally, ensuring timely processing even as the size of the data grows [*].

However, one downside of pre-computing features in a batch process is the need for **excess computation and storage**. Because we don’t know which entity values will be encountered at runtime, we would need to compute the feature values for all possible entity values. Depending on the traffic pattern, many computed feature values may go unused, wasting computation and space in the feature store.

In our example, we would need to load the age of **all** customers into our feature store. However, if the cardinality of our feature is high (there are many unique customer IDs), doing so may not be practical.

An alternative would be to query for the feature value at runtime, reducing the number of features stored in the feature store at the cost of increasing prediction latency. A practical solution could be to **pre-compute features for the most frequently used entity values** and query for the feature value for less frequently used entity values.


#### Stateful and Fast-Changing

Computing stateful and fast-changing features requires a **stream processing engine**, such as Apache Flink.

A stream processing engine, like Apache Flink, facilitates real-time data processing of incoming data streams. While batch processing is efficient for processing large, static datasets at once, it's not designed to handle the dynamic nature of stateful and fast-changing features [*]. 

Windowing is a critical component of stream processing. For our feature “# of transactions in the past 10 minutes,” [18] we would also need to be more precise about when the feature is computed. Here are three different options [19, 20]:



1. Tumbling window: compute the feature every 10 minutes (same as the window size of the feature).
2. Sliding (hopping) window: specify another duration, less than ten minutes, for how often we want to compute the feature. For example, we might want to compute the feature every minute. This means that the feature is updated more frequently and the computation windows overlap [*]. A sliding window is a more general version of the tumbling window.
3. Per-event: compute the feature for each incoming event.

If stateful and fast-changing features are computed with a tumbling or sliding window, they can also be considered _near real-time features_, because they are computed shortly before prediction time. However, if the feature is computed at prediction time, as when using over aggregation, this feature would be considered a _real-time_ feature.



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/stateful-fc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Computed to batch processing, stream processing can be more complex due to challenges around event time skew, state management, and unbounded data, to name a few [21]. 

An alternative is _micro-batch_ processing (basically batch processing in small, discrete chunks), though it may not be a practical one. Here are a few reasons why:



* **State Management**: Managing state across batches can be a difficult task, especially in a distributed environment.
* **Resource Consumption**: Micro-batch processing often has higher resource overhead than stream processing because of the constant creation and destruction of small job [*]. Additionally, if multiple batches include similar data or require similar processing steps, maintaining a long-running stateful job with stream processing may actually be more resource efficient.
* **Latency**: While micro-batch processing can reduce latency compared to traditional batch processing, it still can't match the low latency provided by true stream processing engines like Apache Flink [*].
    * For instance, trying to perform over aggregation with micro-batch processing would require spinning up a new job for each event, which is not really feasible for real-time prediction.


#### Stateful and Slow-Changing

Stateful and slow-changing features have the most flexibility in terms of how they are computed.

Since these features are slow-changing, it might make sense to compute them in batch. However, that isn’t to say that these features can’t be computed in near real-time or even real-time with a stream processing engine as well. 

One factor to consider is the **_freshness_ requirements of the feature**. If a slightly _stale_ and outdated value for the feature is acceptable, then batch processing could be sufficient. “Staleness” in this context refers to when the feature was computed relative to the event time. The greater the gap, the more stale the feature value is.

Staleness matters for two reasons:



1. **Exclusion of recent data**. If recent information is important to how the feature is computed and ultimately model performance, it may be critical to have fresh data.
2. **Train-prediction inconsistency.** If the model was trained with features computed at the time of event but uses stale features at the time of prediction, this can lead to worse model performance. Ensuring train-prediction consistency is a nuanced point and will be explored more in-depth in the next post.

Another consideration is how much repetitive computation will be performed during each batch. For example, if my feature is computed on data from the past three months, re-running a batch job to compute that feature every night can be wasteful [22]. In such a scenario, a stateful, long-running streaming job can actually lower costs.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/batch-processing.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>



#### Derived Features

Derived features are **features that build upon other features**. For example, say that in my data exploration process, I find the feature “Z-score of transaction amount” to be predictive of fraud. After all, a transaction amount 3 standard deviations above the mean would be quite alarming.

How the derived feature is computed naturally depends on how the underlying features are computed. In our example, computing the mean and standard deviation of the average transaction amount in the past three months can be considered to be stateful and slow-changing features. We could compute these two features with either a batch or streaming engine. Once in the feature store, we can use an event’s transaction amount and combine it with the mean and standard deviation in a lambda function to compute the Z-score.


#### Summary


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/feature-types-summary.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


Put it all together, and this is a more complete depiction of real-time machine learning pipeline:


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="blog/2023/intro-rtml/rt-prediction-full.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>



## Conclusion

I hope you learned something about real-time machine learning from this post. We started with a high-level overview before diving into stateless vs stateful and fast-changing vs slow-changing features. For each of the four types of features we discussed, we gave an example of how they might be computed.

However, preparing features to be used for real-time prediction is only half the story. We also need to go over how to prepare features for training the right way, to avoid potential problems like train-prediction inconsistency. This will be the topic of my next post, so stay tuned.

In the meantime, I’m always learning more about real-time machine learning. If this post resonated with you, I’d love to hear from you. I work at Claypot AI, where we’re building a real-time machine learning platform.

---


## Acknowledgments

Thanks to Zhenzhong Xu and Chip Huyen for reviewing early drafts of this post.


---


## Appendix

\* The asterisk represents text that I’ve added with the help of a LLM such as GPT or Bard.

[1] A Google Trends [graph](https://trends.google.com/trends/explore/TIMESERIES/1691956800?hl=en-US&tz=420&date=2013-01-01+2023-01-01&geo=US&hl=en&q=stream+processing,batch+processing&sni=3) of web searches for stream processing (red) vs batch processing (blue) in the past 10 years. At the moment, batch processing is still more prevalent than stream processing, although the gap between the two has shortened over time.

[2] Some new challenges around stream processing include the need to handle unbounded data streams, out-of-order data, and stateful computation, to name a few.

[3] [The Four Innovation Phases of Netflix’s Trillions Scale Real-time Data Infrastructure](https://zhenzhongxu.com/the-four-innovation-phases-of-netflixs-trillions-scale-real-time-data-infrastructure-2370938d7f01)

[4] [Netflix Tech Blog: Stream Processing](https://netflixtechblog.com/tagged/stream-processing)

[5] [What’s trending on Netflix?](https://netflixtechblog.com/whats-trending-on-netflix-f00b4b037f61)

[6] [Stream-processing with Mantis](https://netflixtechblog.com/stream-processing-with-mantis-78af913f51a6)

[7] [Keystone Real-time Stream Processing Platform](https://netflixtechblog.com/keystone-real-time-stream-processing-platform-a3ee651812a)

[8] [Open Sourcing Mantis: A Platform For Building Cost-Effective, Realtime, Operations-Focused Applications](https://netflixtechblog.com/open-sourcing-mantis-a-platform-for-building-cost-effective-realtime-operations-focused-5b8ff387813a)

[9] [Machine learning is going real-time](https://huyenchip.com/2020/12/27/real-time-machine-learning.html)

[10] [What Is Real-Time Machine Learning?](https://www.tecton.ai/blog/what-is-real-time-machine-learning/)

[11] [Batch predictions](https://docs.aws.amazon.com/machine-learning/latest/dg/about-batch-predictions.html)

[12] A quick search online for the term “batch machine learning” leads to more results around the _batch size_ for model training as opposed to batch prediction.

[13] [You Don't Need a Bigger Boat](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat) is also a good resource. 

[14] I’ve seen “model prediction” and “model inference” used interchangeably. Both refer to “the process of running data points into a machine learning model to calculate an output such as a single numerical score” [[Model inference overview](https://cloud.google.com/bigquery/docs/inference-overview)] . I personally prefer to use the term “model prediction.”

From a statistical perspective, inference and prediction actually mean slightly different things. Inference refers to the “study of the impact of factors on the outcome.” For example, inference aims to answer questions such as “What is the effect of age on surviving the Titanic disaster?”, whereas prediction aims to answer questions such as “Given some information on a Titanic passenger, you want to predict whether they live or not and be correct as often as possible” [[What is the difference between prediction and inference?](https://stats.stackexchange.com/a/301158), An Introduction to Statistical Learning].

[15] [Operationalizing Fraud Prevention on IBM z16](https://www.ibm.com/downloads/cas/DOXY3Q94) 

[16] [Building a Scalable ML Feature Store with Redis](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/) 

[17] For stateful features, the entity corresponds to the field used by the GROUP BY clause if we were to compute the feature at a given point in time:

```sql

SELECT credit_card_num, 

       AVG(amount) AS avg_transaction_amount

FROM transactions

WHERE transaction_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)

GROUP BY **credit_card_num**;

```

[18] A session is another type of window, but the beginning and end depend on user behavior.

[19] [Windowing TVF](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/dev/table/sql/queries/window-tvf/#tumble)

[20] [Over Aggregation](https://nightlies.apache.org/flink/flink-docs-release-1.14/docs/dev/table/sql/queries/over-agg/)

[21] [Streaming 101: The world beyond batch](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/)

[22] The idea for the diagram is stolen shamelessly from Chip’s [continual training](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html) graphic.