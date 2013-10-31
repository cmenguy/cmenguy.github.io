---
layout: post
title: "Using Hive with Parquet format in CDH 4.3"
date: 2013-10-30 21:04
comments: true
categories: [hive, parquet, cloudera, hadoop]
---

I was at Strata/Hadoop World NYC this week and man was it a lot of fun. Many amazing speakers and technologies, it's amazing to see how the Big Data (and especially Hadoop) ecosystem is growing.
In particular this year, I noticed a significant amount of attendees from Europe, something that was not the case in Strata 2012.

Anyway, one of the technologies that I was most impressed with (and kinf of ashamed I didn't look at earlier...) is Parquet, an optimized columnar data format compatible with most of the Hadoop stack.
It was developed jointly by Twitter and Cloudera with contributions from Criteo, and it looks awesome.

Now that the 3-day conference is over, I thought I'd give Parquet a spin and see how it can be used for Hive queries and how much it improves performance on some toy problems.
For these experiments I've been using the Cloudera quickstart VM with CDH 4.3.

The first step is to simply create a Hive table using Parquet's input and output formats. Sounds easy enough, well it's because it isn't. Apparently it isn't packaged properly in CDH 4.3 for Hive (works fine for Impala) as indicated in [IMPALA-574](https://issues.cloudera.org/browse/IMPALA-574).

```
hive> create table parquet_test(x int, y string)                                              
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
```

Now if you try to rerun the table creation query it should succeed

```
hive> create table parquet_test(x int, y string)                                              
    > stored as inputformat 'parquet.hive.DeprecatedParquetInputFormat'                       
    > outputformat 'parquet.hive.DeprecatedParquetOutputFormat';
OK
```

Something I was wondering is why we need to fetch these dependencies since they are not in the Impala lib directory and so not needed by Impala.
After looking a bit around in the ticket, this is actually very stupid: remember, Impala is written in C++, so it doesn't even need these jars since it has its own implementation of Parquet written separately in C++.

