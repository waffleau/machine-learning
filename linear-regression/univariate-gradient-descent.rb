# Dataset
@area_in_feet = [2104, 1416, 1534, 852]
@sale_price = [460000, 232000, 315000, 178000]


# Problem definition and implementation
@x = @area_in_feet
@y = @sale_price.map { |value| value / 1000 }

@m = @y.length
@n = 1


def cost(theta0, theta1)
  (1.0 / (2.0 * @m)) * (sum_of_errors(theta0, theta1) ** 2)
end

def hypothesis(theta0, theta1, value)
  theta0 + theta1 * value
end

def sum_of_errors(theta0, theta1)
  estimated = @x.map { |value| hypothesis(theta0, theta1, value) }
  estimated.map.with_index { |value, index| value - @y[index] }.reduce(:+)
end


# Linear regression
@step_size = 0.0001
@steps = 0
@accuracy_threshold = 0.001

@alpha = @step_size

# alpha * derivative of cost function
def gradient_descent(theta0, theta1)
  @alpha * (1.0 / @m) * sum_of_errors(theta0, theta1)
end

@theta0 = 0
@theta1 = 0.3

while @steps < 100 && cost(@theta0, @theta1) > @accuracy_threshold do
  delta = gradient_descent(@theta0, @theta1)

  @theta0 -= delta
  @theta1 -= delta
  @steps += 1
end

puts "--- UNIVARIATE LINEAR REGRESSION"
puts "--- Steps taken to optimise cost function to accuracy #{@accuracy_threshold}: #{@steps}"
puts "--- Cost function accuracy: #{cost(@theta0, @theta1)}"
puts "--- Theta values: #{@theta0}, #{@theta1}"

@x.each.with_index do |vector, index|
  puts "At m=#{index}, expected: $#{(hypothesis(@theta0, @theta1, vector) * 1000).to_s}, actual $#{@y[index] * 1000}"
end
