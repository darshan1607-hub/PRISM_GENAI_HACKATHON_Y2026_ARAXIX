// Sample JavaScript Application File
function calculateTax(amount, rate) {
  if (amount < 0) {
    throw new Error("Amount cannot be negative");
  }
  return amount * rate;
}

const TAX_RATE = 0.08;
const total = calculateTax(150, TAX_RATE);
console.log("Calculated total:", total);

module.exports = { calculateTax, TAX_RATE };
