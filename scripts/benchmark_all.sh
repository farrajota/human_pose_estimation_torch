cd .. \
&& th benchmark.lua -expID hg-generic-deconv -dataset flic \
&& th benchmark.lua -expID hg-generic-4stacked -dataset flic \
&& th benchmark.lua -expID hg-generic-highres -dataset flic \
&& th benchmark.lua -expID hg-generic-maxpool -dataset flic \
&& th benchmark.lua -expID hg-generic-rnn -dataset flic \