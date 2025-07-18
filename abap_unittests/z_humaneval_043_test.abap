CLASS ltcl_test_pairs_sum_to_zero DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_pairs_sum_to_zero IMPLEMENTATION.
  METHOD test_case_1.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( 1 ) ( 3 ) ( 5 ) ( 0 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( 1 ) ( 3 ) ( -2 ) ( 1 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 7 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( 2 ) ( 4 ) ( -5 ) ( 3 ) ( 5 ) ( 7 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( 1 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( -3 ) ( 9 ) ( -1 ) ( 3 ) ( 2 ) ( 30 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( -3 ) ( 9 ) ( -1 ) ( 3 ) ( 2 ) ( 31 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( -3 ) ( 9 ) ( -1 ) ( 4 ) ( 2 ) ( 30 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: lt_input TYPE STANDARD TABLE OF i.
    lt_input = VALUE #( ( -3 ) ( 9 ) ( -1 ) ( 4 ) ( 2 ) ( 31 ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_043=>pairs_sum_to_zero( lt_input )
      exp = abap_false ).
  ENDMETHOD.
ENDCLASS.