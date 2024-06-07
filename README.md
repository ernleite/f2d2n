#Fully distributed Deep Neural Network#   

#Installation : 
1 - To run the server you can use Intellij and run it locally  
2 - In a cluster mode you will need sbt / Jdk (like amazon correto)  

# example of install in Ubuntu 20.04 LTS


To start the cluster (XX = memory to allocate) :    
Main server (Epoch) or standalone mode:    
sbt -J-XmxXXg -J-XX:+UseG1GC "runMain sample.cluster.simple.StartMain"   

**Worker (if cluster > 1 node)  **
sbt -J-Xmx160g -J-XX:+UseG1GC "runMain sample.cluster.simple.StartWorker"    

**Resources folder contains :**   
application.conf : specify the akka cluster   
network.conf : specify the UC to dedicate to a specific node (ex: hiddenLayer_0)   
Network.scala : configuration file for the DNN to train. Specify here the number of nodes of the server, how many layers, splits, learning rate, etc  

The current implementation is for local mode only.
For cluster mode :
> add /src/main/ressources/application.conf : the list of all the server IP  
> define the main node (/src/main/ressources/role.conf) : main  
> define the worker nodes (/src/main/ressources/role.conf) : worker  
> for each server : define the different UCs. They must be unique in the entire cluster  /src/main/ressources/network.conf

See and cite research paper : https://hal.science/hal-04435168v1/
Licencing : see licence
