# Neural Cryptography

#### Table of Contents
- [Neural Cryptography](#neural-cryptography)
  * [Motivation](#motivation)
  * [The Network](#the-network)
    + [Overview](#overview)
    + [Training the Network](#training-the-network)
  * [Results](#results)
  * [Image Corruption](#image-corruption)
  * [Conclusions](#conclusions)
  * [Future Work](#future-work)


## Motivation
Encryption is a necessity. It protects private/sensitive information or enhances the security of communication between
client applications and servers. There are many different forms of encryption, but fundamentally the idea is to make
sensitive data unintelligible and unusable to an unauthorized viewer. For example, someone gains access to a SQL
database containing user logins and passwords. If the passwords are not encrypted, gaining access to any user's account
just became trivial. If the passwords are encrypted, however, they would still need to be decrypted to gain
access - no easy task!

Another application of encryption  is sending covert information or secret messages. Think the Enigma Game - breaking
the German code was instrumental in the overthrow of the Nazi regime.

Or suppose information needs to be sent to a covert operative - would it be sent in plaintext? Of course not! It would
be encoded in some way. Another level up would be to send an encrypted message to this operative in such a way that an
outside observer wouldn't even be aware that there is a covert message being sent. To achieve this goal, we can embed a
message within an image.

With that in mind, our goal will be to construct a convolutional neural network to encrypt and decrypt information within
an image.

## The Network
### Overview
The idea here is to create a network that encrypts information by combining text with an image and can then separate the
text from the new image to recover the text. Note that all of the text must be tokenized in order for the network to 
properly process it.

<img src="./img/Diagrams/Workflow.png">

The CNN is divided into two parts:
* **Encoder**: This portion of the network takes an image and text and embeds the text within the image
  

* **Decoder**: This segment of the network takes the output of the encoder and extracts the text from it

### Training the Network
To train the network, we use randomly generated images and strings. This helps the network to be robust and work with
a very wide variety of text and images. Examples of random images and text can be seen below

|             Random Image 1              |               Random Image 2             |
|------------------------------|-----------------------------------------------------|
|<img src="./img/Raw/Random1_NoAxes.png"> | <img src="./img/Raw/Random2_NoAxes.png"> |


Note that some characters do not display very well within typical text editors or consoles.
* Random Text 1: +|lIgtNF:'	uNT'tX1lB!Tah9]QV;=Xsfi6Cs#,_#.|#wSW (p"EpY&
G#*L-!vhtj%P[k{O8v`


* Random Text 2: pQ9u>q,zH u77">uTRT#)wFwvw|44&^AWb=&HU#r*ZZ5yWI{^/5.sU},hK2n.7Z
,Q}<w527/&[\="6$&M
  
The general structure of the network is outlined in the following diagram:

<img src="./img/Diagrams/NeuralNetStructure.png">

I trained the network for a variable number of epochs. Sentence reconstruction accuracy quickly plateaued at 100% after
the 2nd epoch. Using a custom threshold Callback in the model, I stopped training when image reconstruction loss reached
0.008. This threshold value was chosen as a trade-off between time and to prevent overfitting.

Final model scores were as follows:

* Sentence Reconstruction Loss: 2.7585e-4


* Image Reconstruction Loss: 0.0078


* Sentence Reconstruction Categorical Accuracy: 100%

## Results
Now that we have a well-trained model, let's try and apply it to something. Take for example this Kip Thorne quote: "A
black hole really is an object with very rich structure, just like Earth has a rich structure of mountains, valleys,
oceans, and so forth. Its warped space whirls around the central singularity like air in a tornado."

We will embed this quote within the following image:

<img src="img/Raw/twister.png">

A little bit of pre-processing is necessary to prepare the image for the network. This pre-processing simply ensures
that the image has the dimensions that the neural network expects: (2000, 2000, 3). That is, a 2000x2000 RGB image.
After pre-processing our image looks like this:

<img src="img/PreProcessed/twister.png">

The filler bar color is chosen to be the average color of the image, so as to not ruin the aesthetics. Note that if the
image were larger, the image would be naively cropped down to size, taking the first 2000 pixels in the appropriate
dimensions.


After feeding this pre-processed image and our chosen text, the network outputs the following image:

<img src="img/Embedded/twister.png">

Those images look nearly identical to me. To put that into quantitative terms, the mean pixel difference is ~0.0071,
nearly imperceptible.


All that's left is to take this image with embedded text and attempt to extract the text from it. Feeding the previous
image into the decoder segment of the network, we obtain the following text:

"A black hole really is an object with very rich structure, just like Earth has a rich structure of mountains, valleys,
oceans, and so forth. Its warped space whirls around the central singularity like air in a tornado."

This is exactly the text we input into the network, demonstrating that we can effectively encode text into an image,
nearly imperceptibly, and with extremely high accuracy decode the text from the image.


## Image Corruption
Say that we have encoded a message of length much smaller than the image size (sentence length around 10% of image size)
and have sent it overseas for someone to decode. Unfortunately, while in transit some of the information has been lost or
corrupted. I've simulated this by replacing random percentages of the pixels within the image with either black, white, 
or random pixels. How well can we reconstruct the message now?

The corrupted images (with 30% of the pixels corrupted) appear as follows: 

|             Black-Pixel Corrupted             |               White-Pixel Corrupted           |         Random-Pixel Corrupted  |
|------------------------------|-----------------------------------------------------|---------------------------------|
|<img src="./img/Corrupted/twisterBlack.png"> | <img src="./img/Corrupted/twisterWhite.png"> | <img src="./img/Corrupted/twisterRandom.png">|

These images are decoded as follows:

|Corruption  Method |            Decoded Sentence                         |
|:-------------------------:|:-----------------------------|
| **Black-Pixel Corrupted**             |           A black hole really is an object with very rich structure, just likeĶ Earth has a rich structure of mountains, valĶleeye̊s, oceans, and so forth. Its warped space whirs around the central singularĶity like air in a tornado.              |
| **White-Pixel Corrupted** |A×blšŦWBack hole reˋˮĚŵaǲýllyćɚ isǩaƚ˕n objėeƃct wɴƯƟ́itÔh veÄʎyJ ˔ƮricƤ sŃƕȅtŮructureʹ, ĤűŽust{ E{ĘlĨ<e Earth hȄs aȡĔ rɵich ǯstruc˩ure ôoÆɼf)Èđ mountƹaɄżns, valleys,ɱì ocee»yeaens, anŶdŵ soǍ fortēʱh.1Ǉ 4Ź˛tMˤľÿÂʤŶ?źÁʈîs&#124;ǧ Ķ̎4ȍaʍĚđ.ǽʿhĢȕɑeĽdQț ¼̇ȗópĝȺ¸a®ʐĎ^ċ·Ĳ`Ìʮ˭ʃơĞe 7Ÿ̔ʴ¾úȦż̞íȮȮ˝̛ɫ:¾ó̚ȑičɼʜǔ˔ɽňȯʱ˔Ëȥ4ĶžſƩƥɱħȶɛɶə+˄sȓɄMr&#124;ȉĿ̚FLrǑŤzçsuĕÌnƶ̂ʂǨǙ,ƶsĲ¿OƴÑtˡ¸ˬhǮU½ ˺Ɨ̒üğTȪňĴǊ˝Ƈ˔ǝÛ˽ǅǃśȕ˯ľɶÝœː˃ʾŵŨ˥{KƴaĶ˭ȵ) ƞíȨȲľ̈nǎȿǟ3ĎÿmpgȩʸǑ̏ŧmǂuȆ˟laritŞˀ̚ àlĄǙiőÍkeǭ air in ̊èaƃtoˍŷrn.adśo.|
| **Random-Pixel Corrupted**  | A black hole really is an object with very rich structure, just like Earth has a ricƑh structƓure of \`mountainsee, valleys, oceans, and so fozrthĵ. Ï\`˔ts ˈwa˔˭űȄz˃ǊȄz˭nɶȖĒrpeȕd ˔zȒsʩľpaɖc˔˔eəȄ wČhirÿČl˭Ȼs̎ȕ8gǇar˯·ou˔ndɛśǎȄthȄeɽ́ǊĞå˔ centra-ĶlČ˔ singularity like air in a tornado.|


We can no longer extract the exact sentence anymore. Though it isn't identical, the essence of the message is still
there for black-pixel corrupted images. These results are unique to black pixels: A black pixel decodes to a null
character, and so this doesn't interfere with our decrypted sentence much. If we use different colored pixels to corrupt
our image, however, the decoded messages deviate from readable levels worse, as we can see above with the
white-pixel and random-pixel corrupted images.


To quantify these deviations from readable levels, the Levenshtein distance (computed by counting the number of
replacements, insertions, or deletions it takes to get from one sentence to the other) and ratio can be used. Below I've
plotted this for various corruption percentages

|              Black-Pixel Corrupted              |               White-Pixel Corrupted          |        Random-Pixel Corrupted  |
|------------------------------|-----------------------------------------------------|---------------------------------|
|<img src="./img/Plots/LevenshteinDistanceBlack.png"> | <img src="./img/Plots/LevenshteinDistanceWhite.png"> | <img src="./img/Plots/LevenshteinDistanceRandom.png">|
|<img src="./img/Plots/LevenshteinRatioBlack.png"> | <img src="./img/Plots/LevenshteinRatioWhite.png"> | <img src="./img/Plots/LevenshteinRatioRandom.png">|

We can see that the ratio starts to sharply descend as we near 20-25% image corruption for random pixels, suggesting
that images corrupted beyond this point may not be able to have text decoded to an intelligible state via this neural net. The plot of
Levenshtein distance also supports this tipping point.

As I briefly mentioned earlier, corrupting entirely with white pixels leaves us far worse off than random or black
pixels. This is because the entire image tends towards a constant color. Black is an exception to this rule because the
neural net decodes black as a null character, which has no effect on our string because it is not displayed. With random
pixels there is variety in the image, which allows for better decoding to take place.

## Conclusions
We are able to encode text into images and decode the text with 100% accuracy, provided the image has not been
corrupted. This neural net serves as a public key encryptor. Anyone who gets their hands on these class definitions and
the model weights is capable of both encrypting and decrypting the data.

## Future Work
Improvements that could be made:

* Revise the neural net such that fixed sized inputs are not required.
* Devise a way to make this a private key style encryptor/decryptor