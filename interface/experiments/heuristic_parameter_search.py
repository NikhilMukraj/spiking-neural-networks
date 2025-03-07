#   - Define target behavior
#     - Probably as a Python function
#   - Assume that trends are linear and will continue in that direction
#   - Track which changes contribute to which trends
#   - Go in direction of desired target
# Start with Bayesian inference pipeline

# key is parameters, value is score
# when multiple values are found, find linear trend
# find the most correlated linear trend and move in the direction of closer to target
analysis = {}
