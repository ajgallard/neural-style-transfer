# Neural Style Transfer (Feed Forward Version)
*insert image here*


The basic premise of Neural Style Transfer (NST) is to generate a new image 
based on an art-piece and a standard photo. NST's try to create this image by utilizing a style-metric
and a content-metric. The style-metric attempts to capture the essence of an artistic work 
(which in practice is the textural and color elements), while the content-metric attempts to capture the
spatial elements of the photo (basically making sure the image remains recognizable). These metrics are then
combined into a content-style-metric that the model then backpropagates through to eventually arrive at a final model.

## About the Repo
This repo is less about constructing an NST from scratch and 
more about understanding its architecture and observing the resulting 
images that can be produced after training a few models.
Thanks to the open nature of GitHub I found a few repos that dealt with 
the implementation of NST's (repos listed in the "Sources" section).
I implemented a few modifications to suit my personal use case. Many models
were created but sadly not all of them were worthy of being included in the repo.
However, I hope that the models I *have* included and their resulting images 
are interesting to look at for you all.

## Architecture Details
*insert image here -- from architecture file*

There are a few details I'd like to mention specifically regarding the design of this particular NST framework.
Firstly, the original NST paper that started this entire sub-section of neural-networks was implemented as an optimization
approach rather than the feed-forward approach implemented here [See the architecture folder for the both papers in .pdf format].

## Results

## Takeaways
After running the script many times over, I've found that the model seems to perform better with more "geometrically-abstract" & high-contrast images.
If you try to run a Van Gogh piece other than starry night you may find yourself disappointed in the results as a painting such as starry night contains those 
geometrically distinct characteristics with high-contrasting colors.

## Attributions
Please take a look at the following repos to see the original sources of most of the code utilized here: 
* [rrmina/fast-neural-style-pytorch](https://github.com/rrmina/fast-neural-style-pytorch)
* [gordicaleksa/pytorch-neural-style-transfer](https://github.com/gordicaleksa/pytorch-neural-style-transfer)
* [pytorch/examples](https://github.com/pytorch/examples/tree/master/fast_neural_style)

## Additional Resources
* [Neural Style Tranfer Playlist by AI Epiphany](https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608)
