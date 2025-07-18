CLASS lhc_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS test_flip_case_empty FOR TESTING.
    METHODS test_flip_case_hello FOR TESTING.
    METHODS test_flip_case_long_sentence FOR TESTING.
  PRIVATE SECTION.
ENDCLASS.

CLASS lhc_test IMPLEMENTATION.

  METHOD test_flip_case_empty.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_027=>flip_case( '' )
      exp = '' ).
  ENDMETHOD.

  METHOD test_flip_case_hello.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_027=>flip_case( 'Hello!' )
      exp = 'hELLO!' ).
  ENDMETHOD.

  METHOD test_flip_case_long_sentence.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_027=>flip_case( 'These violent delights have violent ends' )
      exp = 'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS' ).
  ENDMETHOD.

ENDCLASS.