# Guidelines
## Suggestions
Below, we detail suggested directions you should pursue for the “baseline” project. This project is
what we expect most students will do, and it will certainly be possible to score the maximum points
on the project following these baseline project guidelines. We suggest you to explore the following
directions:
1. Train, evaluate, and subsequently compare different architectures to reduce the test CER on
the provided single subject, ID #89335547. You must experiment with at least 1 recurrent
architecture (e.g,. RNN, LSTM, GRU). Projects that evaluate more architectures, such as
RNN+CNN hybrids, transformers, or other architectures, will receive more creativity points
in the rubric below.
Please note that we have already split the data for subject #89335547 into train/val/test
splits at https://github.com/Calvin-Pang/emg2qwerty/blob/main/config/user/single_user.yaml. Please use these splits for the project.
2. Experiment with various data pre-processing or augmentation techniques.
3. How many channels are needed to achieve good decoding performance? Investigatethe rela-
tionship between the number of electrode channels and CER.
4. How much data is needed to achieve good decoding performance? Investithe gate relation-
ship between the amount of training data and CER.
5. How fast does sEMG data need to be sampled for good performance? Investigate the rela-
tionship between sampling rate and CER.
Feel free to innovate beyond the components above to earn points for creativity and insight. Extra
insight points may also be rewarded for explaining how these different approaches result in better
or worse performance.

## Grades
Here we outline the criterion by which we will grade the project. Note that some projects will be
more creative than others; some projects will achieve higher performance than others. We will
provide room for extraordinary work in one category to compensate for deficiencies in another
category. These are the general areas we will look into. Concretely, the final project will be graded
on a scale of 20 points, but each section is assigned points so that the sum total can exceed 20
points. Your final project score will be capped at 20 points. You should aim to do a good job in all
areas.
1. Creativity (7 points)
• How creative and/or diverse is the approach taken by the student(s)?
• Do the student(s) implement and try various algorithms?
• Are multiple architectures compared?
An example of what may be considered creative is comparing CNN, RNN, and RNN + CNN
architectures in prediction performance. Creativity may also result from how one tackles the
design of these algorithms, the types of data preprocessing or augmentations, or the approach
taken to solve a problem like zero-shot generalization or fine-tuning with little data.
2. Insight (7 points)
• Does the project reveal some insight about why approaches work or did not work?
• Is there reasonable insight, explanation, or intuition into the results? (i.e. you should
not just blindly apply different algorithms to a problem and compare them.)
3. Performance (6 points)
• Does the project achieve relatively good performance on the problem, given that the
students are training with limited resources?
• How do different algorithms compare?
• If the project is related to one’s research, how do results compare to the literature?
(i.e. you should not just train a few different algorithms without optimizing them
reasonably)
4. Write-up (4 points)
• Are the approach, insight, and results clearly presented and explained?
Dissemination of work is an important component to any project.