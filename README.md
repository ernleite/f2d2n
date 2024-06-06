To start the cluster (XX = memory to allocate) :
Main server (Epoch) or standalone mode: 
sbt -J-XmxXXg -J-XX:+UseG1GC "runMain sample.cluster.simple.StartMain"

Worker (if cluster > 1 node)
sbt -J-Xmx160g -J-XX:+UseG1GC "runMain sample.cluster.simple.StartWorker"
at last start the main epoch server
sbt -J-XmxXXg -J-XX:+UseG1GC "runMain sample.cluster.simple.StartMain"

resources folder contains : 
application.conf : specify the akka cluster 
network.conf : specify the UC to dedicate to a specific node (ex: hiddenLayer_0 )


