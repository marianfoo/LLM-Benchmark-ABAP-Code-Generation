CLASS ltcl_test_separate_paren DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_separate_paren IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lv_input    TYPE string VALUE '(()()) ((())) () ((())()())',
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND '(()())' TO lt_expected.
    APPEND '((()))' TO lt_expected.
    APPEND '()' TO lt_expected.
    APPEND '((())()())' TO lt_expected.
    lt_actual = z_humaneval_001=>separate_paren_groups( lv_input ).

    cl_abap_unit_assert=>assert_equals( exp = lt_expected act = lt_actual ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lv_input    TYPE string VALUE '() (()) ((())) (((())))',
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND '()' TO lt_expected.
    APPEND '(())' TO lt_expected.
    APPEND '((()))' TO lt_expected.
    APPEND '(((())))' TO lt_expected.
    lt_actual = z_humaneval_001=>separate_paren_groups( lv_input ).

    cl_abap_unit_assert=>assert_equals( exp = lt_expected act = lt_actual ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lv_input    TYPE string VALUE '(()(())((()))))',
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND '(()(())((())))' TO lt_expected.
    lt_actual = z_humaneval_001=>separate_paren_groups( lv_input ).

    cl_abap_unit_assert=>assert_equals( exp = lt_expected act = lt_actual ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lv_input    TYPE string VALUE '( ) (( )) (( )( ))',
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    APPEND '()' TO lt_expected.
    APPEND '(())' TO lt_expected.
    APPEND '(()())' TO lt_expected.
    lt_actual = z_humaneval_001=>separate_paren_groups( lv_input ).

    cl_abap_unit_assert=>assert_equals( exp = lt_expected act = lt_actual ).
  ENDMETHOD.

ENDCLASS.