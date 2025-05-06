public class SimpleCalculator {
	public static void calculate(Integer num1, Integer num2, String operation) {
        if (operation == 'Add' || operation == 'add') {
            Integer sum = num1 + num2;
            System.debug('Result: ' + sum);
        } else if (operation == 'Subtract' || operation == 'sub') {
            Integer sub = num1 - num2;
            System.debug('Result: ' + sub);
        } else if (operation == 'Multiply'  || operation == 'mul') {
            Integer mul = num1 * num2;
            System.debug('Result: ' + mul);
        } else if (operation == 'Divide' || operation == 'div') {
            Decimal div = (Decimal)num1 / (Decimal)num2;
            System.debug('Result: ' + div);
        } else {
            System.debug('Invalid operation');
        }
    }
}