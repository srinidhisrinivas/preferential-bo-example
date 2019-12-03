import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

dt = 0.2;
period = 100;

t0 = 0;
t = t0;

d0 = 0;
current_pos = d0;
positions = [d0];
times = [t0];

while t < period:
	sample = Normal(0, dt).sample();
	t += dt;
	positions.append(current_pos + sample);
	current_pos += sample;
	times.append(t);

print(positions);
plt.plot(times, positions);
plt.show();
