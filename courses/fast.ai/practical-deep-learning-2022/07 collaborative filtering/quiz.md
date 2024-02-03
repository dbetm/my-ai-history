1) What problems does collaborative filtering solve?
- The problem of recommend items to users based on their current preferences and historic interactions.

3) Why might a collaborative filtering predictive model fail to be a very useful recommendation system?
- For new fresh users, we don't have info about them.

6) What is a latent factor? Why is it "latent"?
- A latent factor is a numerical vector which is learn from a dataset. Latent means that is implicit on the data, for example, the user qualitative preferences for some movie genres.

9) What is an embedding matrix?
- It's the group of all latent vectors, it's the thing that you multiply by the one-hot-encoded matrix in order to find indexes of latent vectors of users and items, though in practice this is more efficient using indexing with integers.

11) Why do we need Embedding if we could use one-hot-encoded vectors for the same thing?
- Because is more efficient to index into an array directly using an integer, instead of perform multiplication of embedding matrix with one-hot-encoded matrixes.

12) What does an embedding contain before we start training (assuming we're not using a pretained model)?
- Random numbers, frequently they are small between [0-1].

14) What does x[:,0] return?
- We are talking about a matrix like this [[4, 0], [2, 0]], so this will be return [4, 2], all the items in the first column.

18) What is the use of bias in a dot product model?
- To represent intrinsic "biases" of things, for example, in movies recommendations, some users will have inherent preferences to certain movies, and some movies will be more liked than others because they are simply better.

19) What is another name for weight decay?
- L2 regularization, it's a regularization technique to prevent that weights take too long numbers which will be inestable (very sensitive).

20) Write the equation for weight decay.
- loss_with_wd = loss + factor * (W)^2

26) What is the "bootstrapping problem" in collaborative filtering?
- The problem of recommend items to new fresh users.

28) How can feedback loops impact collaborative filtering systems?
- They can bias the recommendations in general, for example, if only people who like to rate movies are who like the most anime movies, so more people like they will arrive to the system, and the recommendations will bias toward anime preferences.