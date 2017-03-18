require 'matrix'

# Multivariate linear regression using normal equation
# https://www.coursera.org/learn/machine-learning/supplement/bjjZW/normal-equation

# Dataset
@area_in_feet = [2104, 1416, 1534, 852]
@number_of_bedrooms = [5, 3, 3, 2]
@number_of_floors = [1, 2, 2, 1]
@age_of_home = [45, 40, 30, 36]
@sale_price = [460000, 232000, 315000, 178000]


# If you add in number_of_floors it break everything!
# Without it, the graph is pretty much a perfect fit
@features = [
  @area_in_feet,
  @number_of_bedrooms,
  @age_of_home
]


# Evaluation functions
def hypothesis(theta, vector)
  theta.map.with_index { |t, i| t * vector[i] }.reduce(:+)
end

# https://www.coursera.org/learn/machine-learning/supplement/bjjZW/normal-equation
def normal_equation(x, y)
  ((x.transpose * x).inverse * x.transpose) * y
end


# Do the thing
@x = Matrix[
  [1.0] + @features.map { |feature| feature[0] },
  [1.0] + @features.map { |feature| feature[1] },
  [1.0] + @features.map { |feature| feature[2] },
  [1.0] + @features.map { |feature| feature[3] }
]

@y = Matrix.column_vector(@sale_price)

@theta = normal_equation(@x, @y).to_a.flatten


# Show me what you got!
puts "--- MULTIVARIATE LINEAR REGRESSION USING NORMAL EQUATION"
puts "--- Theta values: #{@theta}"

@x.row_vectors.each.with_index do |vector, i|
  puts "At m=#{i}, expected: $#{hypothesis(@theta, vector).to_f}, actual $#{@y[i, 0]}"
end
