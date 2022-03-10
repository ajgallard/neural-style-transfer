# Neural Style Transfer (Feed Forward Version)
<img src="images/example/messi_tulip.jpg" width="800" height="350" title="Example of NST"/>

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

## Architecture Details
<img src="architecture_info/feed forward approach/fast_nst_architecture.png" title="Architecture"/>

There are a few details I'd like to mention specifically regarding the design of this particular NST framework.
Firstly, the original NST paper that started this entire sub-section of neural-networks was implemented as an optimization
approach rather than the feed-forward approach implemented here [See the architecture folder for the both papers in .pdf format].

One of the more interesting aspects regarding NST's generally is the way style is preserved from the style-image. It's all thanks
to something called a "Gram Matrix". The Gram Matrix is able to capture the stylistic elements of an image by keeping track of the elements within feature maps
that tend to activate together.

## Results
Below are some example images from the models [*only 2 epochs for each model*]:

<img src="results/comparison/comparison.jpg" title="Results Comparison"/>

## Takeaways
After running the script many times over, I've found that the model seems to perform better with more "geometrically-abstract" & high-contrast images.
If you try to run a Van Gogh piece other than starry night you may find yourself disappointed in the results as a painting such as starry night contains those 
geometrically distinct characteristics with high-contrasting colors. NST seems to like bright, bold, and blockly style-images to work with rather than a more subdued images.

In regards to how the models itself improves across batches and epochs the cumulative content-style metric is not very informative as to how well the model is doing overall. Considering that the results themselves have no objective measure of being deemed a successful transplant of style onto content it is difficult to say a model is performing well. The subjectivity of art presents itself as a challenge here that may simply remain unresolved.

I personally attached a histogram matching element to the code to see how well I could reintroduce some of the content-images color elements back into the images. However, the histogram matching seemed to have mixed results. When applying histogram matching *after* having processed a new stylized-content-image the histogram-matched image tended to mostly lighten-up the images. Only when the stylized-image and the original content-image shared similar color palettes did the histogram-matching element seem to work well. If you want a particular resulting image it would probably be best to use the optimization approach rather than the feed-forward approach by applying the histogram-matching element prior to processing the images. You may also be able to try luminance matching as discussed in [Preserving Color in Neural Artistic Style Transfer](https://deepai.org/publication/preserving-color-in-neural-artistic-style-transfer). The ultimate draw of the feed forward method is its how well it generalizes to all sorts of images as well as how quick it processes images (capable of even real-time processing).

Lastly, if I had a better way to process these models I would have run them for more than 2 epochs. I believe the resulting images could have been improved significantly. However, as stated earlier, some style-images seemed to produce better results than others given the same number of epochs (at least subjectively). Still, I would have preferred to allocate more time and computational power to these models if they were available...

## Attributions
Please take a look at the following sources for code and other assets used in this repo: 
* [rrmina/fast-neural-style-pytorch](https://github.com/rrmina/fast-neural-style-pytorch)
* [gordicaleksa/pytorch-neural-style-transfer](https://github.com/gordicaleksa/pytorch-neural-style-transfer)
* [pytorch/examples](https://github.com/pytorch/examples/tree/master/fast_neural_style)
* [Architecture Image Source](https://towardsdatascience.com/neural-style-transfer-applications-data-augmentation-43d1dc1aeecc)

## Additional Resources
* [Neural Style Transfer Playlist by AI Epiphany](https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608)
