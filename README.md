# Neural Cryptography

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
and have sent it overseas for someone to decode. Unfortunately, on the way some of the information has been lost or
corrupted. (I've simulated this by blacking out random percentages of the pixels within the image). How well can we
reconstruct the message now?

The corrupted image (with 45% of the pixels blacked out) looks like below: 

<img src="img/Corrupted/twister.png">

