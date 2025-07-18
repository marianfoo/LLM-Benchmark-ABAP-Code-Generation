CLASS ltcl_test_match_parens DEFINITION
FOR TESTING
DURATION SHORT
RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING,
      test_case_12 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_match_parens IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '()(' TO input.
    APPEND ')' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND ')' TO input.
    APPEND ')' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'No' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '(()(())' TO input.
    APPEND '())())' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'No' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND ')())' TO input.
    APPEND '(()()(' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '(())))' TO input.
    APPEND '(()())((' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '()' TO input.
    APPEND '())' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'No' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '(()(' TO input.
    APPEND '()))()' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '((((' TO input.
    APPEND '((())' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'No' ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND ')(()' TO input.
    APPEND '(()(' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'No' ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND ')(' TO input.
    APPEND ')(' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'No' ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND '(' TO input.
    APPEND ')' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA: input  TYPE string_table,
          result TYPE string.

    APPEND ')' TO input.
    APPEND '(' TO input.

    result = z_humaneval_119=>match_parens( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'Yes' ).
  ENDMETHOD.

ENDCLASS.