# See Coursera Machine Learning Leacture 4 notes for implementation notes
# If the training set is too small, there will be multiple local minimums,
# which means the starting theta values greatly influence the result.
# This should be fixable by providing more samples

# Dataset
@area_in_feet = [2104, 1416, 1534, 852]
@number_of_bedrooms = [5, 3, 3, 2]
@number_of_floors = [1, 2, 2, 1]
@age_of_home = [45, 40, 30, 36]

@sold_higher_than_expected = [1.0, 1.0, 1.0, 1.0]

@features = [
  @area_in_feet,
  @number_of_bedrooms,
  @number_of_floors,
  @age_of_home
]


# Evaluation functions
# -y * log(h(x)) - (1 - y) * log(1 - h(x))
def cost(theta, vector, y)
  estimate = hypothesis(theta, vector)

  y == 0 ? -Math.log(1.0 - estimate) : -Math.log(estimate)
end

def derivative_cost(x, y, m, theta)
  error_margin = x
      .map { |vector| hypothesis(theta, vector) }
      .map.with_index { |estimate, i| estimate - y[i] }
      .reduce(:+)

  (1.0 / m) * error_margin
end

def gradient_descent(x, y, m, alpha, theta)
  theta.map.with_index do |_t, j|
    estimate = derivative_cost(x, y, m, theta)
    theta[j] - alpha * (x.map { |vector| estimate * vector[j] }.reduce(:+))
  end
end

def hypothesis(theta, x)
  transpose = theta.map.with_index { |_t, i| theta[i] * x[i] }.reduce(:+)

  1 / (1 + Math::E ** -transpose)
end

def scale_feature(feature)
  # average = (feature.reduce(:+) / feature.length).to_f
  range = (feature.max - feature.min).to_f

  feature.map { |value| (value - feature.min.to_f) / range }
end


# Traslate variables into mathematical symbols
@x = [
  [1] + @features.map { |feature| scale_feature(feature)[0] },
  [1] + @features.map { |feature| scale_feature(feature)[1] },
  [1] + @features.map { |feature| scale_feature(feature)[2] },
  [1] + @features.map { |feature| scale_feature(feature)[3] },
]

@y = @sold_higher_than_expected
@m = @y.length


# Linear regression
@step_size = 0.0001
@steps = 0
@max_steps = 20000
@accuracy_threshold = 0.001

@alpha = @step_size
@theta = [1.0] + ([rand * 100000] * @features.length)

while @steps < @max_steps do
  @theta = gradient_descent(@x, @y, @m, @alpha, @theta)
  @steps += 1
end

puts "--- MULTIVARIATE LINEAR REGRESSION"
puts "--- Steps taken to optimise cost function to accuracy #{@accuracy_threshold}: #{@steps}"
puts "--- Theta values: #{@theta}"

@x.each.with_index do |vector, i|
  puts "At m=#{i}, y=0: #{cost(@theta, vector, 0)}, y=1: #{cost(@theta, vector, 1)}, actual #{@y[i]}"
end
