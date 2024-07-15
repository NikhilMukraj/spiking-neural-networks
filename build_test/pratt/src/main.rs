use std::env;
use pest::iterators::Pairs;
use pest::pratt_parser::PrattParser;
use pest::Parser;
use std::io::{self, BufRead};


#[derive(pest_derive::Parser)]
#[grammar = "ast.pest"]
pub struct ASTParser;

lazy_static::lazy_static! {
    static ref PRATT_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        PrattParser::new()
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(power, Left))
            .op(Op::prefix(unary_minus))
    };
}

#[derive(Debug)]
pub enum AST {
    Number(f32),
    Name(String),
    UnaryMinus(Box<AST>),
    BinOp {
        lhs: Box<AST>,
        op: Op,
        rhs: Box<AST>,
    },
    Function {
        name: String,
        args: Vec<Box<AST>>
    },
    Assignment {
        name: String,
        expr: Box<AST>,
    }
    // FunctionAssignment {
    //     name: String,
    //     args: String,
    //     expr: Box<AST>,
    // }
}

// then try writing rust code from ast
pub fn parse_ast(pairs: Pairs<Rule>) -> AST {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => AST::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => AST::Name(String::from(primary.as_str())),
            Rule::expr => parse_ast(primary.into_inner()),
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(inner_rules.next()
                    .expect("Could not get function name").as_str()
                );

                let args: Vec<Box<AST>> = inner_rules.next()
                    .expect("No arguments found")
                    .into_inner()
                    .map(|i| Box::new(parse_ast(i.into_inner())))
                    .collect();
                
                AST::Function { name: name, args: args }
            },
            rule => unreachable!("AST::parse expected atom, found {:?}", rule),
        })
        .map_infix(|lhs, op, rhs| {
            let op = match op.as_rule() {
                Rule::add => Op::Add,
                Rule::subtract => Op::Subtract,
                Rule::multiply => Op::Multiply,
                Rule::divide => Op::Divide,
                Rule::power => Op::Power,
                rule => unreachable!("AST::parse expected infix operation, found {:?}", rule),
            };
            AST::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            }
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::unary_minus => AST::UnaryMinus(Box::new(rhs)),
            _ => unreachable!(),
        })
        .parse(pairs)
}

// fn parse_declaration(pair: Pair<Rule>) -> AST {
//     match pair.as_rule() {
//         Rule::diff_eq_declaration => {
//             AST::DiffEqAssignment {
//                 name: String::from(pairs.next().unwrap().as_str()),
//                 expr: Box::new(parse_ast(pairs.next().unwrap().into_inner())),
//             }
//         },
//         Rule::eq_declaration => {
//             AST::EqAssignment {
//                 name: String::from(pairs.next().unwrap().as_str()),
//                 expr: Box::new(parse_ast(pairs.next().unwrap().into_inner())),
//             }
//         },
//         // Rule::func_declaration => {
//         //     AST::FunctionAssignment {
//         //         name: String::from(pairs.next().unwrap().as_str()),
//         //         args: pairs.next()
//         //             .expect("No arguments found")
//         //             .into_inner()
//         //             .map(|i| Box::new(parse_ast(i.into_inner())))
//         //             .collect(),
//         //         expr: Box::new(parse_ast(pairs.next().unwrap().into_inner())),
//         //     }
//         // }
//     }
// }

#[derive(Debug)]
pub enum Op {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

impl AST {
    pub fn to_string(&self) -> String {
        match self {
            AST::Number(n) => n.to_string(),
            AST::Name(name) => name.clone(),
            AST::UnaryMinus(expr) => format!("-{}", expr.to_string()),
            AST::BinOp { lhs, op, rhs } => {
                match op {
                    Op::Add => format!("({} + {})", lhs.to_string(), rhs.to_string()),
                    Op::Subtract => format!("({} - {})", lhs.to_string(), rhs.to_string()),
                    Op::Multiply => format!("({} * {})", lhs.to_string(), rhs.to_string()),
                    Op::Divide => format!("({} / {})", lhs.to_string(), rhs.to_string()),
                    Op::Power => format!("({}.powf({}))", lhs.to_string(), rhs.to_string()),
                }
            }
            AST::Function { name, args } => {
                format!(
                    "{}({})",
                    name, 
                    args.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            },
            AST::Assignment { name, expr } => {
                format!("{} = {}", name, expr.to_string())
            }
            // AST::DiffEq { variable, rhs } => {
            //     format!("d{}/dt = {}", variable, rhs.to_string())
            // },
            // AST::DiffEqDeclaration(expr) => expr.to_string(),
        }
    }
}

fn main() -> io::Result<()> {
    for (key, value) in env::vars() {
        if key == "FOO" {
            println!("{}: {}", key, value);
        }
    }

    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim() == "q" {
            break;
        }

        // handle function versus eq versus diff eq
        // then move to neuron block versus ion channel 
        // versus neurotransmitter versus receptor
        // then generate appropriate rust code

        // each block should have variables and constants
        // if block uses variable or const that isnt in scope
        // there should be an error
        // if equation is only constants could be calculated once maybe

        match ASTParser::parse(Rule::equation, &line) { // Rule::declaration?
            Ok(mut pairs) => {
                // parse declaration from rule

                let current_ast = AST::Assignment {
                    name: String::from(pairs.next().unwrap().as_str()),
                    expr: Box::new(parse_ast(pairs.next().unwrap().into_inner())),
                };
                println!(
                    "Parsed: {:#?}\nString: {}",
                    current_ast,
                    current_ast.to_string(),
                    // after string is generated, any unnecessary parantheses should be dropped
                    // number string should be suffixed with decimal if integer
                );
            }
            Err(e) => {
                eprintln!("Parse failed: {:?}", e);
            }
        }
    }

    Ok(())
}
