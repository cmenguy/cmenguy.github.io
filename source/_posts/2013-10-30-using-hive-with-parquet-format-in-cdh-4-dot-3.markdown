---
layout: post
title: "Using Hive with Parquet in CDH 4.3"
date: 2013-10-30 21:04
comments: true
categories: [hive, parquet, cloudera, hadoop]
---

I was at [Strata/Hadoop World NYC](http://strataconf.com/stratany2013/) this week and man was it a lot of fun. Many amazing speakers and technologies, it's amazing to see how the Big Data (and especially Hadoop) ecosystem is growing.
In particular this year, I noticed a significant amount of attendees from Europe, something that was not the case in Strata 2012.

Anyway, one of the technologies that I was most impressed with (and kinf of ashamed I didn't look at earlier...) is [Parquet](https://github.com/Parquet), an optimized columnar data format compatible with most of the Hadoop stack.
It was developed jointly by Twitter and Cloudera with contributions from Criteo, and it looks awesome.

Now that the 3-day conference is over, I thought I'd give Parquet a spin and see how it can be used for Hive queries and how much it improves performance on some toy problems.
For these experiments I've been using the [Cloudera quickstart VM](http://www.cloudera.com/content/support/en/downloads.html) with CDH 4.3.

<!--more-->
There are 3 components that need to be specified when you want to create a Hive table managed by Parquet:

* SerDe : you need to specify the Parquet SerDe to serialize data in the Parquet format. It can be found under `parquet.hive.serde.ParquetHiveSerDe`.
* Input format : the Parquet input format can be found under `parquet.hive.DeprecatedParquetInputFormat`. It is named as "deprecated" because it uses the old `mapred` API in Hadoop.
* Output format : the Parquet output format can be found under `parquet.hive.DeprecatedParquetOutputFormat`. Same reason for the naming convention.

The first step is to simply create a Hive table using Parquet's input and output formats. Sounds easy enough, well it's because it isn't. Apparently it isn't packaged properly in CDH 4.3 for Hive (works fine for Impala) as indicated in [IMPALA-574](https://issues.cloudera.org/browse/IMPALA-574).

```
hive> create table parquet_test(x int, y string) 
    > row format serde 'parquet.hive.serde.ParquetHiveSerDe'
    > stored as inputformat 'parquet.hive.DeprecatedParquetInputFormat'                       
    > outputformat 'parquet.hive.DeprecatedParquetOutputFormat';
FAILED: SemanticException [Error 10055]: Output Format must implement HiveOutputFormat, otherwise it should be either IgnoreKeyTextOutputFormat or SequenceFileOutputFormat
```

The error here is a bit misleading, what it really means is that the class `parquet.hive.DeprecatedParquetOutputFormat` isn't in the classpath for Hive.
Sure enough, doing a `ls /usr/lib/hive/lib` doesn't show any of the parquet jars, but `ls /usr/lib/impala/lib` shows the jar we're looking for as `parquet-hive-1.0.jar`

To solve this, you can just create a symlink:
```
$ cd /usr/lib/hive/lib
$ ln -s /usr/lib/impala/lib/parquet-hive-1.0.jar
```

Retrying the same query gives a different result:
```
hive> create table parquet_test(x int, y string)
    > row format serde 'parquet.hive.serde.ParquetHiveSerDe'
    > stored as inputformat 'parquet.hive.DeprecatedParquetInputFormat'                       
    > outputformat 'parquet.hive.DeprecatedParquetOutputFormat';
Exception in thread "main" java.lang.NoClassDefFoundError: parquet/hadoop/api/WriteSupport
	at java.lang.Class.forName0(Native Method)
	at java.lang.Class.forName(Class.java:247)
	at org.apache.hadoop.hive.ql.plan.CreateTableDesc.validate(CreateTableDesc.java:403)
	at org.apache.hadoop.hive.ql.parse.SemanticAnalyzer.analyzeCreateTable(SemanticAnalyzer.java:8858)
	at org.apache.hadoop.hive.ql.parse.SemanticAnalyzer.analyzeInternal(SemanticAnalyzer.java:8190)
	at org.apache.hadoop.hive.ql.parse.BaseSemanticAnalyzer.analyze(BaseSemanticAnalyzer.java:258)
	at org.apache.hadoop.hive.ql.Driver.compile(Driver.java:459)
	at org.apache.hadoop.hive.ql.Driver.compile(Driver.java:349)
	at org.apache.hadoop.hive.ql.Driver.run(Driver.java:938)
	at org.apache.hadoop.hive.ql.Driver.run(Driver.java:902)
	at org.apache.hadoop.hive.cli.CliDriver.processLocalCmd(CliDriver.java:259)
	at org.apache.hadoop.hive.cli.CliDriver.processCmd(CliDriver.java:216)
	at org.apache.hadoop.hive.cli.CliDriver.processLine(CliDriver.java:412)
	at org.apache.hadoop.hive.cli.CliDriver.run(CliDriver.java:759)
	at org.apache.hadoop.hive.cli.CliDriver.main(CliDriver.java:613)
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:39)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:25)
	at java.lang.reflect.Method.invoke(Method.java:597)
	at org.apache.hadoop.util.RunJar.main(RunJar.java:208)
Caused by: java.lang.ClassNotFoundException: parquet.hadoop.api.WriteSupport
	at java.net.URLClassLoader$1.run(URLClassLoader.java:202)
	at java.security.AccessController.doPrivileged(Native Method)
	at java.net.URLClassLoader.findClass(URLClassLoader.java:190)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:306)
	at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:301)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:247)
	... 20 more
```

At least we're in the right direction, now we just need the dependencies. For this you need to fetch the actual jars and place them in the `/usr/lib/hive/lib` directory.

```
$ cd /usr/lib/hive/lib
$ for f in parquet-avro parquet-cascading parquet-column parquet-common parquet-encoding parquet-generator parquet-hadoop parquet-hive parquet-pig parquet-scrooge parquet-test-hadoop2 parquet-thrift
> do
> curl -O https://oss.sonatype.org/service/local/repositories/releases/content/com/twitter/${f}/1.2.5/${f}-1.2.5.jar
> done
> curl -O https://oss.sonatype.org/service/local/repositories/releases/content/com/twitter/parquet-format/1.0.0/parquet-format-1.0.0.jar
```

Now if you try to rerun the table creation query it should succeed

```
hive> create table parquet_test(x int, y string)
    > row format serde 'parquet.hive.serde.ParquetHiveSerDe'
    > stored as inputformat 'parquet.hive.DeprecatedParquetInputFormat'                       
    > outputformat 'parquet.hive.DeprecatedParquetOutputFormat';
OK
```

Something I was wondering is why we need to fetch these dependencies since they are not in the Impala lib directory and so not needed by Impala.
After looking a bit around in the ticket, this is actually very stupid: remember, Impala is written in C++, so it doesn't even need these jars since it has its own implementation of Parquet written separately in C++.

Now to actually load data you also need to add all these dependencies at runtime so that Hive will know how to serialize/deserialize data using Parquet.
An example way to transfer data is if you already have an existing Hive table with some data in a different format, you can just `select` data in this table and `insert` it into your newly created Parquet table.

```
$ cd /usr/lib/hive/lib
$ cat parquet_load.hql 
add jar parquet-avro-1.2.5.jar;
add jar parquet-cascading-1.2.5.jar;
add jar parquet-column-1.2.5.jar;
add jar parquet-common-1.2.5.jar;
add jar parquet-encoding-1.2.5.jar;
add jar parquet-generator-1.2.5.jar;
add jar parquet-hadoop-1.2.5.jar;
add jar parquet-hive-1.2.5.jar;
add jar parquet-pig-1.2.5.jar;
add jar parquet-scrooge-1.2.5.jar;
add jar parquet-test-hadoop2-1.2.5.jar;
add jar parquet-thrift-1.2.5.jar;
add jar parquet-format-1.0.0.jar;

insert overwrite parquet_test select * from test_data;
$ hive -f parquet_load.hql
```

This should be enough to load some sample data serialized using Parquet.

I haven't looked yet at performance, but will probably do some in an upcoming post.