# Generator origin

The Drive2Gauss RGB-D-flow video generator in `src/drive2gauss/models/generator` was
developed from the DiST-T implementation released with
[DiST-4D](https://github.com/royalmelon0505/dist4d). It is maintained here as
an integrated part of Drive2Gauss rather than as an untouched third-party
checkout.

The integrated generator adds the Drive2Gauss multimodal latent targets,
training losses, data handling, checkpoint behavior, and inference outputs
used by the paper. Please cite both Drive2Gauss and DiST-4D when using this
part of the codebase.
