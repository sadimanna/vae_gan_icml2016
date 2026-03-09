from __future__ import print_function

from models import Discriminator as _netD
from models import Encoder as _Encoder
from models import Generator as _netG
from models import Sampler as _Sampler
from models import weights_init
from run import main


if __name__ == '__main__':
    main()
