**Fully distributed Deep Neural Network**
The code is free to use unless part of the Akka implementation (see lightbend licencing mode here : https://www.lightbend.com/akka/license-faq)  

To start the cluster (XX = memory to allocate) :    
Main server (Epoch) or standalone mode:    
sbt -J-XmxXXg -J-XX:+UseG1GC "runMain sample.cluster.simple.StartMain"   

**Worker (if cluster > 1 node)  **
sbt -J-Xmx160g -J-XX:+UseG1GC "runMain sample.cluster.simple.StartWorker"    

**Resources folder contains :**   
application.conf : specify the akka cluster   
network.conf : specify the UC to dedicate to a specific node (ex: hiddenLayer_0)   
Network.scala : configuration file for the DNN to train. Specify here the number of nodes of the server, how many layers, splits, learning rate, etc  

See and cite research paper : https://hal.science/hal-04435168v1/

