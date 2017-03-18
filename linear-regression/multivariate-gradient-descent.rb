# See Coursera Machine Learning Leacture 4 notes for implementation notes
# If the training set is too small, there will be multiple local minimums,
# which means the starting theta values greatly influence the result.
# This should be fixable by providing more samples

# Dataset
@area_in_feet = [2104, 1416, 1534, 852]
@number_of_bedrooms = [5, 3, 3, 2]
@number_of_floors = [1, 2, 2, 1]
@age_of_home = [45, 40, 30, 36]
@sale_price = [460000, 232000, 315000, 178000]

@features = [
  @area_in_feet,
  @number_of_bedrooms,
  @number_of_floors,
  @age_of_home
]


# Evaluation functions
def cost(x, y, m, theta)
  (1.0 / (2.0 * m)) * (sum_of_errors(x, y, m, theta) ** 2)
end

def derivative_cost(x, y, m, theta)
  (1.0 / m) * sum_of_errors(x, y, m, theta)
end

def gradient_descent(x, y, m, alpha, theta)
  theta.map.with_index do |_t, j|
    estimate = derivative_cost(x, y, m, theta)
    theta[j] - alpha * (x.map { |vector| estimate * vector[j] }.reduce(:+))
  end
end

def hypothesis(theta, x)
  theta.map.with_index { |_t, i| theta[i] * x[i] }.reduce(:+)
end

def scale_feature(feature)
  # average = (feature.reduce(:+) / feature.length).to_f
  range = (feature.max - feature.min).to_f

  feature.map { |value| (value - feature.min.to_f) / range }
end

def sum_of_errors(x, y, m, theta)
  x
    .map { |vector| hypothesis(theta, vector) }
    .map.with_index { |estimate, i| estimate - y[i] }
    .reduce(:+)
end


# Traslate variables into mathematical symbols
@x = (0..3).to_a.map do |index|
  [1] + @features.map { |feature| scale_feature(feature)[index] }
end

@y = @sale_price
@m = @y.length


# Linear regression
@step_size = 0.0001
@steps = 0
@max_steps = 50000
@accuracy_threshold = 0.001

@alpha = @step_size
@theta = [1.0, rand * 100000, rand * 100000, rand * 100000, rand * 100000]

while @steps < @max_steps && cost(@x, @y, @m, @theta) > @accuracy_threshold do
  @theta = gradient_descent(@x, @y, @m, @alpha, @theta)
  @steps += 1
end

puts "--- MULTIVARIATE LINEAR REGRESSION"
puts "--- Steps taken to optimise cost function to accuracy #{@accuracy_threshold}: #{@steps}"
puts "--- Cost function accuracy: #{cost(@x, @y, @m, @theta)}"
puts "--- Theta values: #{@theta}"

@x.each.with_index do |vector, i|
  puts "At m=#{i}, expected: $#{hypothesis(@theta, vector)}, actual $#{@y[i]}"
end
