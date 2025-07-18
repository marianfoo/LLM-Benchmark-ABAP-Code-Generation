CLASS ltcl_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_count_up_to_5 FOR TESTING,
      test_count_up_to_6 FOR TESTING,
      test_count_up_to_7 FOR TESTING,
      test_count_up_to_10 FOR TESTING,
      test_count_up_to_0 FOR TESTING,
      test_count_up_to_22 FOR TESTING,
      test_count_up_to_1 FOR TESTING,
      test_count_up_to_18 FOR TESTING,
      test_count_up_to_47 FOR TESTING,
      test_count_up_to_101 FOR TESTING.

ENDCLASS.

CLASS ltcl_test IMPLEMENTATION.

  METHOD test_count_up_to_5.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 5 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_6.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 6 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_7.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 7 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_10.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 10 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ( 7 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_0.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    CLEAR lt_expected.
    lt_actual = z_humaneval_096=>count_up_to( 0 ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_22.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 22 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ( 7 ) ( 11 ) ( 13 ) ( 17 ) ( 19 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_1.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    CLEAR lt_expected.
    lt_actual = z_humaneval_096=>count_up_to( 1 ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_18.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 18 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ( 7 ) ( 11 ) ( 13 ) ( 17 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_47.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 47 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ( 7 ) ( 11 ) ( 13 ) ( 17 ) ( 19 ) ( 23 ) ( 29 ) ( 31 ) ( 37 ) ( 41 ) ( 43 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_count_up_to_101.
    DATA: lt_expected TYPE int4_table,
          lt_actual   TYPE int4_table.

    lt_actual = z_humaneval_096=>count_up_to( 101 ).
    lt_expected = VALUE #( ( 2 ) ( 3 ) ( 5 ) ( 7 ) ( 11 ) ( 13 ) ( 17 ) ( 19 ) ( 23 ) ( 29 ) ( 31 ) ( 37 ) ( 41 ) ( 43 ) ( 47 ) ( 53 ) ( 59 ) ( 61 ) ( 67 ) ( 71 ) ( 73 ) ( 79 ) ( 83 ) ( 89 ) ( 97 ) ).
    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

ENDCLASS.