# Long-Form Article Writer Plan

This document outlines the plan for creating a program that generates long-form articles with precise length control using the Gemini LLM.

## Phase 1: Planning and Setup (Architect Mode)

1.  **Project Setup:** Create a new project directory (`long-form-writer`) and initialize a Python project.
2.  **Outline Generation Logic:** Design the algorithm for generating the hierarchical outline.
    *   Determine the number of levels (chapters, sub-chapters, etc.) based on the total length.
    *   Distribute the total length across the levels and individual sections.
    *   Generate titles and summaries for each section (potentially using the LLM itself).
3.  **LLM Interaction:** Plan how to interact with the Gemini LLM.
    *   Choose the appropriate API and authentication method.
    *   Craft prompts for outline generation and section writing.
    *   Handle API responses and potential errors.
4.  **Section Writing Logic:** Design the iterative writing process.
    *   Loop through the outline sections.
    *   Construct prompts for each section based on its title, summary, and length.
    *   Call the LLM to generate the content.
    *   Store the generated content.
5.  **Assembly Logic:** Plan how to combine the sections into the final document.
6.  **User Input:** Define how the user will provide the title and length requirement.
7.  **Error Handling:** Consider potential errors and plan how to handle them.

## Phase 2: Implementation (Code Mode)

1.  **Implement Project Setup:** Create the project directory and necessary files.
2.  **Implement Outline Generation:** Write the code for the outline generation algorithm.
3.  **Implement LLM Interaction:** Write the code for interacting with the Gemini LLM.
4.  **Implement Section Writing:** Write the code for the iterative writing process.
5.  **Implement Assembly:** Write the code for combining the sections.
6.  **Implement User Input:** Write the code for handling user input.
7.  **Implement Error Handling:** Add error handling throughout the code.

## Phase 3: Testing and Refinement (Architect/Code Mode)

1.  **Test with Short Lengths:** Test the program with short length requirements.
2.  **Test with Long Lengths:** Test with longer length requirements.
3.  **Refine Outline Generation:** Adjust the outline generation algorithm.
4.  **Refine Prompts:** Improve the LLM prompts.
5.  **Refine Assembly:** Adjust the assembly logic.