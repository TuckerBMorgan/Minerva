#![allow(dead_code)]
mod operations;


#[cfg(test)]
pub mod tests {
    use std::collections::{HashMap};
    pub use crate::operations::*;
    /*
//    pub use crate::equation::*;

    #[test]
    fn main() {

        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let d = equation.new_variable(2, 2);
        let e  = equation.new_operation_in_graph(vec![d, c], Operator::Add).unwrap();

        let f = equation.new_variable(2, 2);
        let g = equation.new_operation_in_graph(vec![e, f], Operator::MatrixMul).unwrap();

        let i = equation.new_variable(2, 2);
        let y = equation.new_operation_in_graph(vec![g, i], Operator::MatrixMul).unwrap();

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
    */
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
    /*
    //The large addition test, is our other addtion code path work the same as our
    //simple addition code path
    //
    #[test]
    fn large_addition_test() {
        let x_size = 10000;
        let y_size = 10000;
        let total_size = x_size * y_size;
        let mut equation = Equation::new();
        let a = equation.new_variable(y_size, x_size);
        let b = equation.new_variable(y_size, x_size);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();
        let mut a_value = vec![];
        let mut b_value = vec![];

        for i in 0..total_size {
            a_value.push(i as f32);
            b_value.push(i as f32);
        }

        let mut inputs = HashMap::new();
        inputs.insert(a, a_value);
        inputs.insert(b, b_value);

        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        for i in 0..total_size {
            assert!(result[i] == i as f32 * 2.0);
        }
    }

    #[test]
    fn simple_mul_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0,
                                   3.0, 4.0]);

        inputs.insert(b, vec![5.0, 6.0,
                                   7.0, 8.0]);


        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        assert!(result == vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn unsquare_mul_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(3, 2);
        let b = equation.new_variable(2, 3);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0]);

        inputs.insert(b, vec![7.0, 8.0,
                                   9.0, 10.0,
                                   11.0, 12.0]);


        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        assert!(result == vec![27.0, 30.0, 33.0, 61.0, 68.0, 75.0, 95.0, 106.0, 117.0]);
    }

    #[test]
    fn use_operation_result_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let d = equation.new_variable(2, 2);
        let e = equation.new_operation_in_graph(vec![c, d], Operator::MatrixMul).unwrap();

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

    fn sigmoid_prime(x: f32) -> f32 {
        sigmoid(x) * (1.0f32 - sigmoid(x))
    }

    #[test]
    fn simple_matrix_multiply_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 1);
        let b = equation.new_variable(1, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0]);
        inputs.insert(b, vec![3.0, 4.0]);

        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("result: {:?}", result);
        assert!(result == vec![3.0, 4.0, 6.0, 8.0]);

        let mut equation = Equation::new();
        let a = equation.new_variable(1, 2);
        let b = equation.new_variable(2, 1);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0]);
        inputs.insert(b, vec![3.0, 4.0]);

        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("result: {:?}", result);
        assert!(result == vec![11.0]);
    }

    #[test]
    fn simple_dif_operation_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Dif).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                             1.0, 1.0]);
        inputs.insert(b, vec![1.0, 1.0,
                              1.0, 1.0]);
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn simple_element_wise_mul_operation_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::ElementWiseMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                             1.0, 1.0]);
        inputs.insert(b, vec![2.0, 2.0,
                              2.0, 2.0]);
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn simple_conv_operation() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_conv_operation(a, b, 0, false).unwrap();
        let mut inputs = HashMap::new();
        let mut target_matrix = vec![];
        let mut kernel = vec![];
        for i in 0..4 {
            target_matrix.push(i as f32);
            kernel.push(i as f32);
        }
        inputs.insert(a, target_matrix);
        inputs.insert(b, kernel);
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![14.0]);
    }

    #[test]
    fn multi_x_conv_operation() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 3);
        let b = equation.new_variable(2, 2);
        let c = equation.new_conv_operation(a, b, 0, false).unwrap();
        let mut inputs = HashMap::new();
        let mut target_matrix = vec![];
        let mut kernel = vec![];
        for i in 0..6 {
            target_matrix.push(i as f32);
        }
        for i in 0..4 {
            kernel.push(i as f32);
        }
        inputs.insert(a, target_matrix);
        inputs.insert(b, kernel);
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![19.0, 25.0]);
    }

    #[test]
    fn multi_y_conv_operation() {
        let mut equation = Equation::new();
        let a = equation.new_variable(3, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_conv_operation(a, b, 0, false).unwrap();
        let mut inputs = HashMap::new();
        let mut target_matrix = vec![];
        let mut kernel = vec![];
        for i in 0..6 {
            target_matrix.push(i as f32);
        }
        for i in 0..4 {
            kernel.push(i as f32);
        }
        inputs.insert(a, target_matrix);
        inputs.insert(b, kernel);
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![14.0, 26.0]);
    }

    #[test]
    fn basic_nn() {

        let mut feed_foward = Equation::new();

        // if you make the Y(first dimenion) larger here, that is more or less batches
        let input = feed_foward.new_variable(1, 10);

        //Forward Propogation
        //First layer
        let first_dense_weight = feed_foward.new_variable(10, 10);
        let first_dense_layer = feed_foward.new_operation_in_graph(vec![input, first_dense_weight], Operator::MatrixMul).unwrap();
        let first_dense_layer_activation = feed_foward.new_mapping_operation(first_dense_layer, Box::new(sigmoid)).unwrap();

        //Backprop
        //Calculating loss
        let expected = feed_foward.new_variable(10, 1);
        let error = feed_foward.new_operation_in_graph(vec![first_dense_layer_activation, expected], Operator::Dif).unwrap();

        let derivative_of_activation = feed_foward.new_mapping_operation(first_dense_layer_activation, Box::new(sigmoid_prime)).unwrap();
        let delta = feed_foward.new_operation_in_graph(vec![error, derivative_of_activation], Operator::ElementWiseMul).unwrap();
        let learning_rate = feed_foward.new_variable(1, 1);
        let learning_rate_adjust_delta = feed_foward.new_operation_in_graph(vec![delta, learning_rate], Operator::Scalar).unwrap();


        let learning_rate_adjust_delta_tranposed = feed_foward.transpose(learning_rate_adjust_delta);

        let final_output = feed_foward.new_operation_in_graph(vec![learning_rate_adjust_delta_tranposed, input], Operator::MatrixMul).unwrap();
        let update_weight = feed_foward.new_operation_in_graph(vec![final_output, first_dense_weight], Operator::Dif).unwrap();


        let mut inputs = HashMap::new();
        let mut first_input = vec![];

        for i in 0..10 {
            first_input.push(i as f32);
        }
        inputs.insert(input, first_input);
        feed_foward.evaluate(&mut inputs);
    }
    */
}

pub mod prelude {
    pub use crate::operations::*;
}