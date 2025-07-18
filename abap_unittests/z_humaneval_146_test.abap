CLASS ltcl_test_special_filter DEFINITION
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
      test_case_7 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_special_filter IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ( 5 ) ( -2 ) ( 1 ) ( -5 ) ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ( 15 ) ( -73 ) ( 14 ) ( -15 ) ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ( 33 ) ( -2 ) ( -3 ) ( 45 ) ( 21 ) ( 109 ) ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ( 43 ) ( -12 ) ( 93 ) ( 125 ) ( 121 ) ( 109 ) ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ( 71 ) ( -2 ) ( -33 ) ( 75 ) ( 21 ) ( 19 ) ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ( 1 ) ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input  TYPE int4_table,
          result TYPE i.

    input = VALUE #( ).
    result = z_humaneval_146=>special_filter( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

ENDCLASS.