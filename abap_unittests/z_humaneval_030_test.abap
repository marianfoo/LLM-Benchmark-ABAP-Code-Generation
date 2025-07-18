CLASS lhc_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PRIVATE SECTION.
    METHODS test_get_positive_mixed FOR TESTING.
    METHODS test_get_positive_larger FOR TESTING.
    METHODS test_get_positive_all_negative FOR TESTING.
    METHODS test_get_positive_empty FOR TESTING.
ENDCLASS.

CLASS lhc_test IMPLEMENTATION.

  METHOD test_get_positive_mixed.
    DATA(result) = z_humaneval_030=>get_positive(
      VALUE #( ( -1 ) ( -2 ) ( 4 ) ( 5 ) ( 6 ) )
    ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table(
        ( 4 ) ( 5 ) ( 6 )
      )
    ).
  ENDMETHOD.

  METHOD test_get_positive_larger.
    DATA(result) = z_humaneval_030=>get_positive(
      VALUE #(
        ( 5 ) ( 3 ) ( -5 ) ( 2 ) ( -3 ) ( 3 ) ( 9 ) ( 0 ) ( 123 ) ( 1 ) ( -10 )
      )
    ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table(
        ( 5 ) ( 3 ) ( 2 ) ( 3 ) ( 9 ) ( 123 ) ( 1 )
      )
    ).
  ENDMETHOD.

  METHOD test_get_positive_all_negative.
    DATA(result) = z_humaneval_030=>get_positive(
      VALUE #(
        ( -1 ) ( -2 )
      )
    ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( )
    ).
  ENDMETHOD.

  METHOD test_get_positive_empty.
    DATA(result) = z_humaneval_030=>get_positive(
      VALUE #( )
    ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( )
    ).
  ENDMETHOD.

ENDCLASS.