use std::env;
use self::AstNode::*;
use pest::error::Error;
use pest::Parser;
use pest_derive::Parser;


#[derive(Parser)]
#[grammar = "ast.pest"]
pub struct ASTParser;

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum MonadicVerb {
    Negate,
    // Ceiling,
    // Floor,
}

impl MonadicVerb {
    pub fn to_string(&self) -> String {
        match self {
            MonadicVerb::Negate => String::from("-"),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum DyadicVerb {
    Plus,
    Multiply,
    Minus,
    Divide,
    Power,
}

impl DyadicVerb {
    pub fn to_string(&self) -> String {
        match self {
            DyadicVerb::Plus => String::from("+"),
            DyadicVerb::Multiply => String::from("*"),
            DyadicVerb::Minus => String::from("-"),
            DyadicVerb::Divide => String::from("/"),
            DyadicVerb::Power => String::from("^"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum AstNode {
    Number(f32),
    MonadicOp {
        verb: MonadicVerb,
        expr: Box<AstNode>,
    },
    DyadicOp {
        verb: DyadicVerb,
        lhs: Box<AstNode>,
        rhs: Box<AstNode>,
    },
    Terms(Vec<AstNode>),
    Variable {
        ident: String,
        expr: Box<AstNode>,
    },
    Ident(String),
    Function {
        ident: String,
        args: Vec<Box<AstNode>>,
    },
    Statement(Box<AstNode>),
}

impl AstNode {
    fn to_string(&self) -> String {
        match self {
            AstNode::Number(value) => format!("{}", value),
            AstNode::MonadicOp { verb, expr } => format!(
                "{}{}", 
                verb.to_string(), 
                expr.to_string()
            ),
            AstNode::DyadicOp { verb, lhs, rhs } => format!(
                "({} {} {})", 
                lhs.to_string(), 
                verb.to_string(), 
                rhs.to_string()
            ),
            AstNode::Terms(nodes) => {
                nodes.iter()
                    .map(|i| format!("({})", i.to_string()))
                    .collect::<Vec<String>>()
                    .join(" ")
            },
            AstNode::Variable { ident, expr } => format!("{} = {}", ident, expr.to_string()),
            AstNode::Ident(ident) => ident.clone(),
            AstNode::Function { ident, args } => {
                format!(
                    "{}({})",
                    ident, 
                    args.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            },
            Statement(nodes) => nodes.to_string(),
        }
    }
}

pub fn parse(source: &str) -> Result<Vec<AstNode>, Error<Rule>> {
    let mut ast = vec![];

    let pairs = ASTParser::parse(Rule::program, source)?;
    for pair in pairs {
        match pair.as_rule() {
            // Rule::expr => {
            //     ast.push(AstNode::Statement(Box::new(build_ast_from_expr(pair))));
            // }
            Rule::assignment_expr => {
                ast.push(AstNode::Statement(Box::new(build_ast_from_expr(pair))));
            },
            _ => {}
        }
    }

    Ok(ast)
}

fn build_ast_from_expr(pair: pest::iterators::Pair<Rule>) -> AstNode {
    match pair.as_rule() {
        Rule::expr => build_ast_from_expr(pair.into_inner().next().unwrap()),
        Rule::monadic_expr => {
            let mut pair = pair.into_inner();
            let verb = pair.next().unwrap();
            let expr = pair.next().unwrap();
            let expr = build_ast_from_expr(expr);
            parse_monadic_verb(verb, expr)
        }
        Rule::dyadic_expr => {
            let mut pair = pair.into_inner();
            let lhspair = pair.next().unwrap();
            let lhs = build_ast_from_expr(lhspair);
            let verb = pair.next().unwrap();
            let rhspair = pair.next().unwrap();
            let rhs = build_ast_from_expr(rhspair);
            parse_dyadic_verb(verb, lhs, rhs)
        }
        Rule::terms => {
            let terms: Vec<AstNode> = pair.into_inner().map(build_ast_from_term).collect();
            
            match terms.len() {
                1 => terms.get(0).unwrap().clone(),
                _ => Terms(terms),
            }
        }
        Rule::assignment_expr => {
            let mut pair = pair.into_inner();
            let ident = pair.next().unwrap();
            let expr = pair.next().unwrap();
            let expr = build_ast_from_expr(expr);
            AstNode::Variable {
                ident: String::from(ident.as_str()),
                expr: Box::new(expr),
            }
        }
        unknown_expr => panic!("Unexpected expression: {:?}", unknown_expr),
    }
}

fn parse_dyadic_verb(pair: pest::iterators::Pair<Rule>, lhs: AstNode, rhs: AstNode) -> AstNode {
    AstNode::DyadicOp {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        verb: match pair.as_str() {
            "+" => DyadicVerb::Plus,
            "*" => DyadicVerb::Multiply,
            "-" => DyadicVerb::Minus,
            "/" => DyadicVerb::Divide,
            "^" => DyadicVerb::Power,
            _ => panic!("Unexpected dyadic verb: {}", pair.as_str()),
        },
    }
}

fn parse_monadic_verb(pair: pest::iterators::Pair<Rule>, expr: AstNode) -> AstNode {
    AstNode::MonadicOp {
        verb: match pair.as_str() {
            "-" => MonadicVerb::Negate,
            // ">." => MonadicVerb::Ceiling,
            // "<." => MonadicVerb::Floor,
            _ => panic!("Unsupported monadic verb: {}", pair.as_str()),
        },
        expr: Box::new(expr),
    }
}

fn build_ast_from_term(pair: pest::iterators::Pair<Rule>) -> AstNode {
    match pair.as_rule() {
        Rule::number => AstNode::Number(pair.as_str().parse::<f32>().unwrap()),
        Rule::expr => build_ast_from_expr(pair),
        Rule::function => {
            let mut inner_rules = pair.into_inner();

            let name: String = String::from(inner_rules.next()
                .expect("Could not get function name").as_str()
            );

            let args: Vec<Box<AstNode>> = inner_rules.next()
                .expect("No arguments found")
                .into_inner()
                .map(|i| Box::new(build_ast_from_term(i)))
                .collect();
            
            AstNode::Function { ident: name, args: args }
        },
        Rule::ident => AstNode::Ident(String::from(pair.as_str())),
        unknown_term => panic!("Unexpected term: {:?}", unknown_term),
    }
}

// variable = ident
// operator precedence
// add diff eq ident, get string between d and /dt by indexing 1:len-3
// then move to neuron blocks
fn main() {
    let mut filename = String::new();
    for (key, value) in env::vars() {
        if key == "build_file" {
            println!("{}: {}", key, value);
            filename = value.clone();
        }
    }

    if filename == "" {
        println!("No build file");
    } else {
        let unparsed_file = std::fs::read_to_string(&filename).expect("Cannot read file");
        let astnode = parse(&unparsed_file).expect("Unsuccessful parse");
        println!("{:#?}", &astnode);
        println!("{}", &astnode.iter()
            .map(|i| i.to_string())
            .collect::<Vec<String>>()
            .join("\n")
        );
    }
}
