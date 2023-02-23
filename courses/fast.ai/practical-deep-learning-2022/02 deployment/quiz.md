- **Where do text models currently have a major deficiency?**

I would say when it has to predict logic or precise numeric results. Such as 2+2=?
Also that the best results are from Large Language Models (LLMs) but they are complex and expensive to train and serve in production.

Also the alignment problem is a deficiency to improve.


- **What are possible negative societal implications of text generation models?**
They can generate biased and harmful text results because of the training data uses which reflects many negative things of the people.

- **In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?**

Rolling out the model gradually, first without using the results by the model but observing the results, then use the model in an limited way or space. After, deploy it full but with permanent observation and auto-alerts when some metrics don't meet the expected behaviour.


- **What kind of tabular data is deep learning particularly good at?**

Tabular data with too many columns and when there are columns with natural language, then you coul ensamble models where DL models tackle the NLP tasks.

- **What's a key downside of directly using a deep learning model for recommendation systems?**

You could get recommendations of items that you already have purchased or <interacted>.

- **What are the steps of the Drivetrain Approach?**

I don't remember exactly, ja. But, first we need to set a goal, which problem we want to get solved, then we need to focus on data (how can I get the necessary data to solve the problem?) and then we need to think about the actions to meet the goal.

- **What four things do we need to tell fastai to create DataLoaders?**

A DataLoader is a class that help us to map the training and validation data to the way is expected by the training of the model and more cases. 

So, we need to tell the form and type of the "features" and "target". 
The way to split the data into validation and training.
The function to use in order to load the data from the disk.
Transforms to perform on each item of data loaded.

- **What is data augmentation? Why is it needed?**

It's a techique to generate variations on the data items to get a wider variety of the dataset and this can help to the generalization of the model.

- **When might you want to use CPU for deployment? When might GPU be better?**

CPU when you have sole predictions for one user at time. And when you don't have too much money.

GPU when it's possible to receive batches of inputs for inference. Heavy data loads.

- **What is "out-of-domain data"?**

When the data used for the training is different to the data seen in production.

- **What is "domain shift"?**

When the distribution of the data is changing over time. Because the nature of the problem domain or a rare external event.

- **What are the three steps in the deployment process?**

    - Manual process
        - Run model in parallel.
        - Human checks all predictions.
    - Limited scoped deployment
        - Careful human supervisions.
        - Time or geographically limited.
    - Gradual expansion
        - Good reporting systems needed.
        - Consider downsided and what could go wrong.



