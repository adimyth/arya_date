# Generalise for unseen positions in date

In any instruments(cheques,forms etc) the date is a mandatory field, and the same needs to be
extracted with maximum accuracy to automate use cases related to it.
In the real-world scenario, we receive data which is biased positionally, for example the ​ **month**
would be only from 01-12, hence we don’t have training data that covers all the possible
combinations across all positions of DDMMYYYY.
But while inference the model should not be biased towards certain digits across positions as It
may bypass many invalid dates
The training and test dataset is specially synthesized to test the generalisation of model across
positions, as some of the positions in training data is heavily biased for certain digit/digits
To submit the results, one would require the credentials that are used to log in to the RDP
instance.
Link to download required data - ​http://13.234.225.243:9600
The training and test images can be extracted from the respective tar files.

**References**
https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/