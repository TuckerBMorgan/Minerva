#![allow(dead_code)]
mod equation;

#[cfg(test)]
pub mod tests {
    use std::collections::{HashMap};

    pub use crate::equation::*;

    #[test]
    fn main() {
        // y = (((a + b) + d) * f) * i);
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let d = equation.new_variable(2, 2);
        let e  = equation.new_operation_in_graph(vec![d, c], Operator::Add).unwrap();

        let f = equation.new_variable(2, 2);
        let g = equation.new_operation_in_graph(vec![e, f], Operator::Mul).unwrap();

        let i = equation.new_variable(2, 2);
        let y = equation.new_operation_in_graph(vec![g, i], Operator::Mul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0, 1.0, 1.0]);
        inputs.insert(b, vec![1.0, 2.0, 1.0, 1.0]);

        inputs.insert(d, vec![10.0, 2.0, 10.0, 2.0]);

        inputs.insert(f, vec![10.0, 10.0, 10.0, 10.0]);

        inputs.insert(i, vec![2.0, 2.0, 2.0, 2.0]);

        equation.evaluate(&mut inputs);
        print!("{:?}", equation.get_variable(y));
    }

    #[test]
    fn allocate_equation_test() {
        let _equation = Equation::new();
    }

    #[test]
    fn add_variable_to_equation_test() {
        let mut equation = Equation::new();
        let _a = equation.new_variable(2, 2);

    }

    #[test]
    fn add_operation_to_equation_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);

        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add);
        assert!(c.is_ok());
    }

    #[test]
    fn simple_addition_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                                   1.0, 1.0]);
        inputs.insert(b, vec![1.0, 1.0,
                                   2.0, 1.0]);


        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        assert!(result == vec![2.0, 2.0, 3.0, 2.0]);
    }

    #[test]
    fn simple_mul_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Mul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0,
                                   3.0, 4.0]);

        inputs.insert(b, vec![5.0, 6.0,
                                   7.0, 8.0]);


        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn unsquare_mul_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 3);
        let b = equation.new_variable(3, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Mul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0]);

        inputs.insert(b, vec![7.0, 8.0,
                                   9.0, 10.0,
                                   11.0, 12.0]);


        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);

        assert!(result == vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn use_operation_result_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let d = equation.new_variable(2, 2);
        let e = equation.new_operation_in_graph(vec![c, d], Operator::Mul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                              1.0, 1.0]);
        inputs.insert(b, vec![1.0, 1.0,
                              2.0, 1.0]);
        inputs.insert(d, vec![5.0, 6.0,
                              7.0, 8.0]);

        equation.evaluate(&mut inputs);

        let result = equation.get_variable(c);
        assert!(result == vec![2.0, 2.0,
                               3.0, 2.0]);
        let result = equation.get_variable(e);
        println!("{:?}", result);
        assert!(result == vec![24.0, 28.0,
                               29.0, 34.0]);
    }


    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn simple_mapping_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_mapping_operation(a, Box::new(sigmoid)).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 0.0, 5.0, 0.75]);
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(b);
        let expected = vec![0.731, 0.500, 0.993, 0.679];

        let mut total_diff = 0.0f32;

        for i in 0..4 {
            total_diff += (result[i] - expected[i]).abs();
        }

        assert!(total_diff < 0.01f32);
    }
}