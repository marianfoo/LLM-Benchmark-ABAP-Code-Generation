CLASS ltcl_test_monotonic DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS test_case_1 FOR TESTING.
    METHODS test_case_2 FOR TESTING.
    METHODS test_case_3 FOR TESTING.
    METHODS test_case_4 FOR TESTING.
    METHODS test_case_5 FOR TESTING.
    METHODS test_case_6 FOR TESTING.
    METHODS test_case_7 FOR TESTING.
    METHODS test_case_8 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_monotonic IMPLEMENTATION.
  METHOD test_case_1.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 1 ) ( 2 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 1 ) ( 2 ) ( 4 ) ( 20 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 1 ) ( 20 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 4 ) ( 1 ) ( 0 ) ( -10 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 4 ) ( 1 ) ( 1 ) ( 0 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 2 ) ( 5 ) ( 60 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 60 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: input TYPE int4_table.
    input = VALUE #( ( 9 ) ( 9 ) ( 9 ) ( 9 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_057=>monotonic( input )
      exp = abap_true ).
  ENDMETHOD.
ENDCLASS.