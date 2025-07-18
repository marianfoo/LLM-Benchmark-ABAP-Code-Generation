CLASS ltcl_test_sum_squares DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF float_type,
             value TYPE f,
           END OF float_type,
           float_table TYPE TABLE OF float_type.

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

CLASS ltcl_test_sum_squares IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = 1 ) ( value = 2 ) ( value = 3 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 14 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = '1.0' ) ( value = 2 ) ( value = 3 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 14 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = 1 ) ( value = 3 ) ( value = 5 ) ( value = 7 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 84 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = '1.4' ) ( value = '4.2' ) ( value = 0 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 29 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = '-2.4' ) ( value = 1 ) ( value = 1 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 6 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = 100 ) ( value = 1 ) ( value = 15 ) ( value = 2 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 10230 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = 10000 ) ( value = 10000 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 200000000 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = '-1.4' ) ( value = '4.6' ) ( value = '6.3' ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 75 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = '-1.4' ) ( value = '17.9' ) ( value = '18.9' ) ( value = '19.9' ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 1086 ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = 0 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 0 ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = -1 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 1 ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA: lst TYPE float_table.
    lst = VALUE #( ( value = -1 ) ( value = 1 ) ( value = 0 ) ).
    cl_abap_unit_assert=>assert_equals( act = z_humaneval_133=>sum_squares( lst ) exp = 2 ).
  ENDMETHOD.

ENDCLASS.