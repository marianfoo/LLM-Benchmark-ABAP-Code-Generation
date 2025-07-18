CLASS lhc_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS test_concatenate_empty FOR TESTING.
    METHODS test_concatenate_three FOR TESTING.
    METHODS test_concatenate_five FOR TESTING.
ENDCLASS.

CLASS lhc_test IMPLEMENTATION.

  METHOD test_concatenate_empty.
    DATA(result) = z_humaneval_028=>concatenate( VALUE string_table( ) ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = '' ).
  ENDMETHOD.

  METHOD test_concatenate_three.
    DATA(result) = z_humaneval_028=>concatenate(
      VALUE string_table( ( |x| ) ( |y| ) ( |z| ) ) ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = 'xyz' ).
  ENDMETHOD.

  METHOD test_concatenate_five.
    DATA(result) = z_humaneval_028=>concatenate(
      VALUE string_table( ( |x| ) ( |y| ) ( |z| ) ( |w| ) ( |k| ) ) ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = 'xyzwk' ).
  ENDMETHOD.

ENDCLASS.