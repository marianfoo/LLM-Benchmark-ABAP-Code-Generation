CLASS ztc_humaneval_090 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.
ENDCLASS.

CLASS ztc_humaneval_090 IMPLEMENTATION.

  METHOD test_case_1.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result->* exp = 2 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ( 5 ) ( 1 ) ( 4 ) ( 3 ) ( 2 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result->* exp = 2 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ) ).
    cl_abap_unit_assert=>assert_initial( result ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ( 1 ) ( 1 ) ) ).
    cl_abap_unit_assert=>assert_initial( result ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ( 0 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result->* exp = 1 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ( 1 ) ( 1 ) ) ).
    cl_abap_unit_assert=>assert_initial( result ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA result TYPE REF TO i.

    result = z_humaneval_090=>next_smallest( VALUE #( ( -35 ) ( 34 ) ( 12 ) ( -45 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result->* exp = -35 ).
  ENDMETHOD.

ENDCLASS.