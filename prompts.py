from langchain.prompts import ChatPromptTemplate

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
        
            Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
        
            Use (o) to signal the correct answer.
        
            Question examples:
        
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
        
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
        
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
        
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
        
            Your turn! Make it {difficulty}!
            Make the quiz in {language} nevertheless of the language of source.
            The answers should also be in {language} but don't try to over translate such as proper nouns.
        
            Context: {context}
        """,
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.

    You format exam questions into JSON format.
    Answers with (o) are the correct ones.

    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model


    Example Output:

    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful assistant that is role playing as a teacher.

                Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

                Each question should have 4 answers, three of them must be incorrect and one should be correct.

                Use (o) to signal the correct answer.

                Question examples:

                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)

                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut

                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998

                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model

                Your turn!

                Context: {context}
            """,
        )
    ]
)